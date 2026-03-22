"""
subway_rt.py - Real-time MTA GTFS-RT subway delay ingestion.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
_DSN = os.getenv("TIMESCALE_DSN")
_MTA_API_KEY = os.getenv("MTA_API_KEY", "")
_BASE_URL = "https://api.mta.info/GTFS"
_FEED_IDS = [1, 2, 11, 16, 21, 26, 31, 36]

# Map stop_id -> zone_id
_ZONE_MAP: dict[str, int] = {}


def load_zone_map(csv_path: str = "pulsecast/data/stop_to_zone.csv"):
    global _ZONE_MAP
    if not _ZONE_MAP:
        try:
            df = pd.read_csv(csv_path)
            df["stop_id"] = df["stop_id"].astype(str)
            _ZONE_MAP = dict(zip(df["stop_id"], df["zone_id"]))
            logger.info("Loaded %d stop-to-zone mappings.", len(_ZONE_MAP))
        except Exception as e:
            logger.error("Failed to load stop_to_zone map: %s", e)


def fetch_feed(feed_id: int) -> list[dict]:
    """Fetch GTFS-RT feed and parse TripUpdate delays."""
    headers = {"x-api-key": _MTA_API_KEY} if _MTA_API_KEY else {}
    url = f"{_BASE_URL}?feed_id={feed_id}"
    
    delays = []
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        
        for entity in feed.entity:
            if entity.HasField('trip_update'):
                for stu in entity.trip_update.stop_time_update:
                    if stu.HasField('arrival') and stu.arrival.HasField('delay'):
                        delays.append({
                            "stop_id": str(stu.stop_id),
                            "delay": float(stu.arrival.delay)
                        })
    except Exception as e:
        logger.warning("Failed to fetch feed %d: %s", feed_id, e)
    
    return delays


def process_delays(feed_id: int, delays: list[dict], current_hour: datetime) -> pd.DataFrame:
    """Aggregate stop-level delays to zone-level mean delay."""
    if not delays:
        return pd.DataFrame(columns=["zone_id", "hour", "feed_id", "mean_delay", "trip_count"])
        
    df = pd.DataFrame(delays)
    df["zone_id"] = df["stop_id"].map(_ZONE_MAP)
    df = df.dropna(subset=["zone_id"])
    
    if df.empty:
        return pd.DataFrame(columns=["zone_id", "hour", "feed_id", "mean_delay", "trip_count"])
    
    df["zone_id"] = df["zone_id"].astype(int)
    
    agg = df.groupby("zone_id").agg(
        mean_delay=("delay", "mean"),
        trip_count=("delay", "count")
    ).reset_index()
    
    agg["hour"] = current_hour
    agg["feed_id"] = str(feed_id)
    return agg


def write_to_db(df: pd.DataFrame, dsn: str | None):
    """Upsert aggregated delays into the subway_delay table."""
    if df.empty or dsn is None:
        return

    conn = psycopg2.connect(dsn)
    try:
        params = [
            (
                int(row["zone_id"]),
                row["hour"],
                row["feed_id"],
                float(row["mean_delay"]),
                int(row["trip_count"])
            )
            for _, row in df.iterrows()
        ]

        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO subway_delay (zone_id, hour, feed_id, mean_delay, trip_count)
                    VALUES %s
                    ON CONFLICT (zone_id, hour, feed_id)
                    DO UPDATE SET 
                        mean_delay = (subway_delay.mean_delay * subway_delay.trip_count + EXCLUDED.mean_delay * EXCLUDED.trip_count) / (subway_delay.trip_count + EXCLUDED.trip_count),
                        trip_count = subway_delay.trip_count + EXCLUDED.trip_count;
                    """,
                    params,
                )
        logger.info("Upserted %d zone aggregations to DB.", len(params))
    finally:
        conn.close()


def poll():
    load_zone_map()
    
    backoff = 1
    while True:
        start_time = time.time()
        current_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        
        all_success = True
        for feed_id in _FEED_IDS:
            delays = fetch_feed(feed_id)
            if delays:
                agg_df = process_delays(feed_id, delays, current_hour)
                if not agg_df.empty:
                    try:
                        write_to_db(agg_df, _DSN)
                        backoff = 1  # Reset backoff on successful DB write
                    except Exception as e:
                        logger.error("Database error while writing feed %d: %s", feed_id, e)
                        all_success = False

        elapsed = time.time() - start_time
        
        if not all_success:
            backoff = min(backoff * 2, 300)
            logger.warning("Errors encountered. Backing off for %d seconds.", backoff)
            time.sleep(backoff)
        else:
            sleep_time = max(0, 60 - elapsed)
            time.sleep(sleep_time)


def main():
    logger.info("Starting GTFS-RT subway poller...")
    poll()


if __name__ == "__main__":
    main()
