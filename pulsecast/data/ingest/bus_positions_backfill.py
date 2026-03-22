"""
bus_positions_backfill.py - Historical GTFS-RT bus positions ingestion.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import pandas as pd
import psycopg2

from pulsecast.data.ingest.bus_positions import _DSN, _get_taxi_zones, process_date

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def get_processed_days(start_date: date, end_date: date, dsn: str | None) -> set[date]:
    """Query the DB for days that already have congestion data."""
    if dsn is None:
        return set()
    
    try:
        conn = psycopg2.connect(dsn)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT date_trunc('day', hour)::date FROM congestion WHERE hour >= %s AND hour <= %s",
                    (start_date, end_date + timedelta(days=1))
                )
                rows = cur.fetchall()
                return {row[0] for row in rows}
    except Exception as e:
        logger.warning("Failed to query existing days: %s", e)
        return set()
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()


def backfill(start_date: date, end_date: date, dsn: str | None = _DSN):
    logger.info("Starting bus positions backfill from %s to %s", start_date, end_date)
    
    zones = _get_taxi_zones()
    processed_days = get_processed_days(start_date, end_date, dsn)
    
    current_date = start_date
    days_processed_in_run = 0
    
    while current_date <= end_date:
        if current_date in processed_days:
            logger.info("Skipping %s, already present in DB", current_date)
        else:
            logger.info("Processing %s", current_date)
            # process_date handles fetch, variance computation, and upsert
            process_date(current_date, zones, dsn)
        
        current_date += timedelta(days=1)
        days_processed_in_run += 1
        
        if days_processed_in_run % 10 == 0:
            logger.info("Progress: processed %d days (currently at %s)", days_processed_in_run, current_date - timedelta(days=1))
            
    logger.info("Backfill complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date()
    
    backfill(start_date, end_date)


if __name__ == "__main__":
    main()
