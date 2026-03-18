"""
tlc.py – Downloads the last 24 months of NYC TLC Yellow and Green taxi
Parquet files from the public nyc.gov/tlc data repository, filters to the
four columns of interest, and aggregates to hourly pickup counts per zone.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import psycopg2
import requests

logger = logging.getLogger(__name__)

_DSN: str | None = os.getenv("TIMESCALE_DSN")

# Zone-to-route mapping: TLC PULocationID → demand.route_id.
# When no explicit mapping is present the zone ID is used directly as the route ID.
_ZONE_TO_ROUTE: dict[int, int] = {}

# Base URL pattern used by the TLC open-data programme.
_TLC_BASE = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
)

_KEEP_COLS = ["pickup_datetime", "PULocationID", "trip_distance", "fare_amount"]

# Column name aliases differ between Yellow and Green trip record schemas.
_COL_ALIASES: dict[str, dict[str, str]] = {
    "yellow": {
        "tpep_pickup_datetime": "pickup_datetime",
    },
    "green": {
        "lpep_pickup_datetime": "pickup_datetime",
    },
}


def _month_range(months: int = 24) -> list[tuple[int, int]]:
    """Return a list of (year, month) tuples covering the last *months* months."""
    today = date.today().replace(day=1)
    result: list[tuple[int, int]] = []
    for _ in range(months):
        today = (today - timedelta(days=1)).replace(day=1)
        result.append((today.year, today.month))
    return result


def download_parquet(
    color: str,
    year: int,
    month: int,
    dest_dir: Path,
) -> Path | None:
    """Download a single TLC Parquet file; returns *None* on HTTP error."""
    url = _TLC_BASE.format(color=color, year=year, month=month)
    dest = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
    if dest.exists():
        logger.info("Already cached: %s", dest)
        return dest
    logger.info("Downloading %s …", url)
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        logger.warning("HTTP %d for %s – skipping", resp.status_code, url)
        return None
    dest.write_bytes(resp.content)
    return dest


def load_and_filter(path: Path, color: str) -> pl.DataFrame:
    """Read a Parquet file, rename columns, and keep only the four target columns."""
    df = pl.read_parquet(path)
    aliases = _COL_ALIASES.get(color, {})
    rename_map = {k: v for k, v in aliases.items() if k in df.columns}
    if rename_map:
        df = df.rename(rename_map)
    available = [c for c in _KEEP_COLS if c in df.columns]
    df = df.select(available)
    # Ensure pickup_datetime is a proper datetime.
    if "pickup_datetime" in df.columns:
        df = df.with_columns(
            pl.col("pickup_datetime").cast(pl.Datetime("us"))
        )
    return df


def aggregate_hourly(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to hourly pickup counts per TLC zone (PULocationID)."""
    df = df.with_columns(
        pl.col("pickup_datetime").dt.truncate("1h").alias("hour")
    )
    return (
        df.group_by(["PULocationID", "hour"])
        .agg(pl.len().alias("pickup_count"))
        .sort(["PULocationID", "hour"])
    )


def write_to_db(df: pl.DataFrame, dsn: str) -> int:
    """Upsert aggregated hourly counts into the *demand* hypertable.

    Maps ``PULocationID`` to ``route_id`` via :data:`_ZONE_TO_ROUTE`, falling
    back to the raw zone ID when no explicit mapping exists.

    Returns the number of rows written.
    """
    if df.is_empty():
        return 0

    params = [
        (
            _ZONE_TO_ROUTE.get(int(r["PULocationID"]), int(r["PULocationID"])),
            r["hour"],
            int(r["pickup_count"]),
        )
        for r in df.iter_rows(named=True)
    ]

    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO demand (route_id, hour, volume)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (route_id, hour)
                    DO UPDATE SET volume = EXCLUDED.volume;
                    """,
                    params,
                )
    finally:
        conn.close()

    return len(params)


def ingest(
    dest_dir: Path = Path("data/raw/tlc"),
    months: int = 24,
    colors: tuple[str, ...] = ("yellow", "green"),
    dsn: str | None = _DSN,
) -> pl.DataFrame:
    """
    Download the last *months* months of TLC data for each *color*,
    aggregate to hourly counts, persist to TimescaleDB when *dsn* is provided,
    and return a combined Polars DataFrame.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []

    for year, month in _month_range(months):
        for color in colors:
            path = download_parquet(color, year, month, dest_dir)
            if path is None:
                continue
            try:
                df = load_and_filter(path, color)
                hourly = aggregate_hourly(df)
                frames.append(hourly)
            except Exception:
                logger.exception("Failed to process %s", path)

    if not frames:
        return pl.DataFrame(
            schema={"PULocationID": pl.Int64, "hour": pl.Datetime("us"), "pickup_count": pl.UInt32}
        )

    combined = (
        pl.concat(frames)
        .group_by(["PULocationID", "hour"])
        .agg(pl.col("pickup_count").sum())
        .sort(["PULocationID", "hour"])
    )

    if dsn is not None:
        written = write_to_db(combined, dsn)
        logger.info("Wrote %d rows to demand table.", written)

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ingest()
    print(result)
