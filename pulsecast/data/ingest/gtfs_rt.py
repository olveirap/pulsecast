"""
gtfs_rt.py – Polls the MTA GTFS-Realtime trip-updates feed every 60 s and
computes a *delay_index* (mean arrival delay weighted by trip count) per
TLC zone and truncated hour.  Results are written to TimescaleDB.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import psycopg2
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

load_dotenv()
_MTA_FEED_URL = os.getenv("MTA_FEED_URL", "https://api.mta.info/GTFS")
_MTA_API_KEY = os.getenv("MTA_API_KEY", "")

_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast",
)

_DEFAULT_STOP_ZONE_MAP = Path(__file__).resolve().parents[1] / "stop_to_zone.csv"


def _resolve_stop_zone_map_path(path: str | Path | None = None) -> Path:
    """Resolve the stop-zone CSV path from explicit arg, env var, or default."""
    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        cleaned = path.strip()
        if cleaned:
            return Path(cleaned)

    env_path = (os.getenv("STOP_ZONE_MAP_PATH") or "").strip()
    if env_path:
        return Path(env_path)

    return _DEFAULT_STOP_ZONE_MAP


def _load_stop_to_zone(path: str | Path | None = None) -> dict[str, int]:
    """Load stop_id -> TLC zone mapping from CSV.

    Expected CSV columns: stop_id,zone_id.
    """
    csv_path = _resolve_stop_zone_map_path(path)
    if not csv_path.exists():
        logger.warning(
            "Stop-to-zone CSV not found at %s. delay_index rows will be empty until map is generated.",
            csv_path,
        )
        return {}
    if csv_path.is_dir():
        logger.warning(
            "Stop-to-zone CSV path points to a directory (%s). delay_index rows will be empty.",
            csv_path,
        )
        return {}

    mapping: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"stop_id", "zone_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid stop-zone CSV at {csv_path}: expected columns {sorted(required)}"
            )

        for row in reader:
            stop_id = (row.get("stop_id") or "").strip()
            zone_id_raw = (row.get("zone_id") or "").strip()
            if not stop_id or not zone_id_raw:
                continue
            try:
                mapping[stop_id] = int(zone_id_raw)
            except ValueError:
                logger.debug("Skipping row with non-integer zone_id: %s", row)

    logger.info("Loaded %d stop-to-zone mappings from %s", len(mapping), csv_path)
    return mapping


_STOP_TO_ZONE: dict[str, int] = _load_stop_to_zone()


def _fetch_feed() -> gtfs_realtime_pb2.FeedMessage:
    """Fetch and parse the MTA GTFS-Realtime protobuf feed."""
    headers: dict[str, str] = {}
    if _MTA_API_KEY:
        headers["x-api-key"] = _MTA_API_KEY
    resp = requests.get(_MTA_FEED_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    return feed


def _compute_delay_index(
    feed: gtfs_realtime_pb2.FeedMessage,
) -> list[dict]:
    """
    Compute mean arrival delay weighted by trip count, grouped by
    (zone_id, truncated_hour).

    Returns a list of dicts with keys: zone_id, hour, delay_index.
    """
    zone_delays: dict[tuple[int, datetime], list[float]] = {}

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        trip_update = entity.trip_update
        for stop_time_update in trip_update.stop_time_update:
            stop_id = stop_time_update.stop_id
            zone_id = _STOP_TO_ZONE.get(stop_id)
            if zone_id is None:
                continue

            if stop_time_update.HasField("arrival"):
                delay_sec = stop_time_update.arrival.delay
            elif stop_time_update.HasField("departure"):
                delay_sec = stop_time_update.departure.delay
            else:
                continue

            # Use feed timestamp to determine the hour bucket.
            ts = datetime.fromtimestamp(feed.header.timestamp, tz=UTC)
            hour = ts.replace(minute=0, second=0, microsecond=0)
            key = (zone_id, hour)
            zone_delays.setdefault(key, []).append(delay_sec)

    rows = []
    for (zone_id, hour), delays in zone_delays.items():
        if not delays:
            continue
        delay_index = sum(delays) / len(delays)
        rows.append(
            {
                "zone_id": zone_id,
                "hour": hour,
                "delay_index": delay_index,
                "trip_count": len(delays),
            }
        )
    return rows


def _upsert_rows(rows: list[dict]) -> None:
    """Write delay_index rows to TimescaleDB using an upsert."""
    if not rows:
        return
    conn = psycopg2.connect(_DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO delay_index (zone_id, hour, delay_index, disruption_flag)
                        VALUES (%(zone_id)s, %(hour)s, %(delay_index)s, FALSE)
                        ON CONFLICT (zone_id, hour)
                        DO UPDATE SET delay_index = EXCLUDED.delay_index;
                        """,
                        row,
                    )
    finally:
        conn.close()


def poll() -> None:
    """Single poll cycle: fetch → parse → upsert."""
    logger.info("Polling MTA GTFS-RT feed …")
    try:
        feed = _fetch_feed()
        rows = _compute_delay_index(feed)
        _upsert_rows(rows)
        logger.info("Upserted %d delay_index rows.", len(rows))
    except Exception:
        logger.exception("Error during GTFS-RT poll cycle")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    scheduler = BlockingScheduler()
    scheduler.add_job(poll, "interval", seconds=60, id="gtfs_rt_poll")
    logger.info("Starting GTFS-RT poller (interval=60 s) …")
    scheduler.start()


if __name__ == "__main__":
    main()
