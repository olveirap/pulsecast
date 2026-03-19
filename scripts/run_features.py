"""
run_features.py – Feature engineering pipeline for Pulsecast.

Loads raw demand and congestion data from TimescaleDB, runs feature modules,
and materializes the result as a Parquet snapshot for training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl
import psycopg2

from pulsecast.features.calendar import build_calendar_features
from pulsecast.features.congestion import build_congestion_features
from pulsecast.features.demand import build_demand_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast",
)
_OUT_DIR = Path(os.getenv("FEATURES_DIR", "data/features"))


def load_from_timescaledb() -> pl.DataFrame:
    """Fetch demand and congestion variance from TimescaleDB and join them."""
    logger.info("Connecting to TimescaleDB at %s", _DB_DSN)
    
    conn = psycopg2.connect(_DB_DSN)
    
    # We use a LEFT JOIN to ensure we have all demand rows even if congestion
    # is missing for some (zone, hour) pairs.
    query = """
    SELECT 
        d.route_id, 
        d.hour, 
        d.volume, 
        COALESCE(c.travel_time_var, 0.0) as delay_index,
        COALESCE(c.travel_time_var, 0.0) as travel_time_var,
        COALESCE(c.sample_count, 0) as sample_count
    FROM demand d
    LEFT JOIN congestion c ON d.route_id = c.zone_id AND d.hour = c.hour
    ORDER BY d.route_id, d.hour
    """
    
    try:
        # Polars can read from a DB connection directly
        df = pl.read_database(query, conn)
        logger.info("Loaded %d rows from database.", len(df))
        return df
    finally:
        conn.close()


def main() -> None:
    out_dir = Path(os.getenv("FEATURES_DIR", "data/features"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    try:
        df = load_from_timescaledb()
    except Exception as e:
        logger.error("Failed to load data from TimescaleDB: %s", e)
        logger.warning("Pipeline cannot proceed without source data.")
        raise SystemExit(1) from e

    if df.is_empty():
        logger.warning("No data found in database. Run 'make ingest' first.")
        raise SystemExit(1)

    # 2. Run feature engineering
    logger.info("Building demand features...")
    df = build_demand_features(df)
    
    logger.info("Building calendar features...")
    df = build_calendar_features(df)
    
    logger.info("Building congestion features...")
    # build_congestion_features expects 'zone_id', but we have 'route_id'
    df = df.rename({"route_id": "zone_id"})
    df = build_congestion_features(df)
    df = df.rename({"zone_id": "route_id"})

    # 3. Materialize
    out_path = out_dir / "features_latest.parquet"
    df.write_parquet(out_path)
    logger.info("Features materialized to %s", out_path)


if __name__ == "__main__":
    main()
