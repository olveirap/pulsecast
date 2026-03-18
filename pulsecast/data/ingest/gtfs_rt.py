"""
gtfs_rt.py – Polls the MTA GTFS-Realtime trip-updates feed every 60 s and
computes a *delay_index* (mean arrival delay weighted by trip count) per
TLC zone and truncated hour.  Results are written to TimescaleDB.
"""

from __future__ import annotations

import logging
import os
import datetime as dt
from datetime import datetime

import psycopg2
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from google.transit import gtfs_realtime_pb2

logger = logging.getLogger(__name__)

_MTA_FEED_URL = os.getenv("MTA_FEED_URL", "https://api.mta.info/GTFS")
_MTA_API_KEY = os.getenv("MTA_API_KEY", "")

_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast",
)

# MTA stop_id → TLC zone mapping (truncated demo subset; extend as needed).
# In production this should be loaded from a lookup table in the database.
_STOP_TO_ZONE: dict[str, int] = {
    # A few sample entries; a real deployment would load all ~500 zones.
    "101": 1,
    "102": 1,
    "201": 2,
    "301": 3,
}


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
            ts = datetime.fromtimestamp(feed.header.timestamp, tz=dt.UTC)
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
