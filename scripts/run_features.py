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


def load_from_timescaledb() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fetch demand, routes, and congestion data from TimescaleDB."""
    logger.info("Connecting to TimescaleDB at %s", _DB_DSN)
    conn = psycopg2.connect(_DB_DSN)
    try:
        df_demand = pl.read_database("SELECT route_id, hour, volume FROM demand", conn)
        df_routes = pl.read_database(
            "SELECT route_id, origin_zone_id, destination_zone_id FROM routes", conn
        )
        df_congestion = pl.read_database(
            "SELECT zone_id, hour, travel_time_var, sample_count FROM congestion", conn
        )
        return df_demand, df_routes, df_congestion
    finally:
        conn.close()


def main() -> None:
    out_dir = Path(os.getenv("FEATURES_DIR", "data/features"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    try:
        df_demand, df_routes, df_congestion = load_from_timescaledb()
    except Exception as e:
        logger.error("Failed to load data from TimescaleDB: %s", e)
        raise SystemExit(1) from e

    if df_demand.is_empty():
        logger.warning("No demand data found. Run 'make ingest' first.")
        raise SystemExit(1)

    # 2. Run feature engineering
    logger.info("Building demand features...")
    df = build_demand_features(df_demand)

    logger.info("Building calendar features...")
    df = build_calendar_features(df)

    logger.info("Building congestion features...")
    # Compute features at the zone level first
    df_cong_feat = build_congestion_features(df_congestion)

    # Resolve route_id to zones and join features twice
    df = df.join(df_routes, on="route_id", how="inner")

    # Join origin features
    origin_cols = {
        c: f"origin_{c}"
        for c in df_cong_feat.columns
        if c not in ["zone_id", "hour"]
    }
    df = df.join(
        df_cong_feat.rename(origin_cols),
        left_on=["origin_zone_id", "hour"],
        right_on=["zone_id", "hour"],
        how="left",
    )

    # Join destination features
    dest_cols = {
        c: f"dest_{c}"
        for c in df_cong_feat.columns
        if c not in ["zone_id", "hour"]
    }
    df = df.join(
        df_cong_feat.rename(dest_cols),
        left_on=["destination_zone_id", "hour"],
        right_on=["zone_id", "hour"],
        how="left",
    )

    # Fill missing congestion with 0
    df = df.fill_null(0.0)

    # 3. Materialize
    out_path = out_dir / "features_latest.parquet"
    df.write_parquet(out_path)
    logger.info("Features materialized to %s", out_path)


if __name__ == "__main__":
    main()
