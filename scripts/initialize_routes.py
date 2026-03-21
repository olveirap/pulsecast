"""
initialize_routes.py – Identifies high-volume OD pairs and populates the routes table.

Scans the last 3 available months of TLC data, calculates average monthly volume
per (PULocationID, DOLocationID), and inserts pairs with >1000 trips/month into
the routes reference table.
"""

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast",
)
_DATA_DIR = Path("data/raw/tlc")
_THRESHOLD = 1000


def _month_range(months: int = 3) -> list[tuple[int, int]]:
    today = date.today().replace(day=1)
    result = []
    for _ in range(months):
        today = (today - timedelta(days=1)).replace(day=1)
        result.append((today.year, today.month))
    return result


def main():
    if not _DATA_DIR.exists():
        logger.error("Data directory %s not found.", _DATA_DIR)
        return

    frames = []
    unique_months = set()  # Track unique months for which data was found
    
    # 1. Scan last 3 months of data
    for year, month in _month_range(3):
        for color in ["yellow", "green"]:
            path = _DATA_DIR / f"{color}_tripdata_{year}-{month:02d}.parquet"
            if path.exists():
                logger.info("Loading %s", path)
                df = pl.read_parquet(path)
                # Keep only PU and DO
                df = df.select([
                    pl.col("PULocationID").cast(pl.Int32),
                    pl.col("DOLocationID").cast(pl.Int32)
                ])
                frames.append(df)
                # Track the unique month (year, month) tuple
                unique_months.add((year, month))

    if not frames:
        logger.warning("No TLC parquet files found in %s for the last 3 months.", _DATA_DIR)
        return

    # 2. Calculate average monthly volume using actual number of months with data
    combined = pl.concat(frames)
    total_months = len(unique_months)  # Dynamic count based on actual data availability
    
    # Log warning if fewer than expected months of data are available
    if total_months < 3:
        logger.warning(
            "Only %d month(s) of data found (expected 3). "
            "Average volume calculation may not be representative.",
            total_months
        )
    
    # Edge case: if no unique months found (shouldn't happen given the check above, but safe to handle)
    if total_months == 0:
        logger.error("No months with data found. Cannot calculate average volume.")
        return
    
    route_stats = (
        combined.group_by(["PULocationID", "DOLocationID"])
        .len()
        .with_columns((pl.col("len") / total_months).alias("avg_monthly_volume"))
        .filter(pl.col("avg_monthly_volume") > _THRESHOLD)
        .sort("avg_monthly_volume", descending=True)
    )

    logger.info("Found %d routes meeting the threshold of %d trips/month.", len(route_stats), _THRESHOLD)

    if route_stats.is_empty():
        return

    # 3. Populate routes table
    try:
        with psycopg2.connect(_DB_DSN) as conn:
            with conn.cursor() as cur:
                # We use INSERT ... ON CONFLICT DO NOTHING to avoid duplicates
                params = [
                    (int(r["PULocationID"]), int(r["DOLocationID"]))
                    for r in route_stats.iter_rows(named=True)
                ]
                from psycopg2.extras import execute_values
                execute_values(cur, """
                    INSERT INTO routes (origin_zone_id, destination_zone_id)
                    VALUES %s
                    ON CONFLICT (origin_zone_id, destination_zone_id) DO NOTHING
                """, params)
            logger.info("Successfully populated routes table.")
    except Exception as e:
        logger.error("Error populating routes table: %s", e)


if __name__ == "__main__":
    main()
