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
from pulsecast.features.demand import build_demand_features, build_duration_features

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
        df_demand = pl.read_database("SELECT route_id, hour, volume, avg_duration FROM demand", conn)
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

    logger.info("Building duration features...")
    df = build_duration_features(df)

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

    # --- Data Quality Checks ---
    logger.info("Running data quality checks...")
    
    # 1. Assert no all-null columns for congestion features
    if len(df) > 100:
        congestion_cols = [c for c in df.columns if "delay_index" in c or "disruption_flag" in c or "low_confidence_flag" in c]
        for col in congestion_cols:
            null_count = df.select(pl.col(col).null_count()).item()
            if null_count == len(df):
                raise AssertionError(f"Column {col} is all-null!")

    # 2. Assert no leakage: origin_delay_index_lag1 at time T equals travel_time_var at time T-1
    if len(df) > 100:
        leakage_check = df.select(["origin_zone_id", "hour", "origin_delay_index_lag1"]).drop_nulls()
        df_cong_shifted = df_congestion.with_columns(
            (pl.col("hour") + pl.duration(hours=1)).alias("hour")
        )
        leakage_join = leakage_check.join(
            df_cong_shifted,
            left_on=["origin_zone_id", "hour"],
            right_on=["zone_id", "hour"],
            how="inner"
        )
        if len(leakage_join) > 0:
            diffs = leakage_join.filter(
                (pl.col("origin_delay_index_lag1") - pl.col("travel_time_var")).abs() > 1e-6
            )
            if len(diffs) > 0:
                raise AssertionError(f"Leakage detected! {len(diffs)} rows have mismatched lags.")

    # 3. Assert disruption_flag has non-zero variance
    if len(df) > 200 and "origin_disruption_flag" in df.columns:
        var_origin = df.select(pl.col("origin_disruption_flag").var()).item()
        if var_origin is None or var_origin == 0:
            logger.warning("origin_disruption_flag has zero variance or is null.")

    # 4. Data Quality Report
    print("\n" + "="*40)
    print("      DATA QUALITY REPORT")
    print("="*40)
    print(f"Total Rows: {len(df)}")
    
    print("\n1. Row count per route (top 5):")
    for row in df.group_by("route_id").agg(pl.len().alias("count")).sort("count", descending=True).head(5).iter_rows():
        print(f"  Route {row[0]}: {row[1]} rows")

    print("\n2. Min/Max of key features:")
    key_features = ["volume", "origin_delay_index_lag1", "dest_delay_index_lag1", "origin_disruption_flag", "dest_disruption_flag"]
    for kf in key_features:
        if kf in df.columns:
            min_val = df.select(pl.col(kf).min()).item()
            max_val = df.select(pl.col(kf).max()).item()
            print(f"  {kf}: min={min_val}, max={max_val}")

    print("\n3. Null rates (after 168h warm-up):")
    df_warm = df.with_columns(
        (pl.col("hour") - pl.col("hour").min().over("route_id")).alias("time_from_start")
    ).filter(pl.col("time_from_start") >= pl.duration(hours=168))

    if len(df_warm) > 0:
        null_rates = df_warm.select([
            (pl.col(c).null_count() / pl.len()).alias(c) for c in df_warm.columns if c != "time_from_start"
        ]).row(0)
        
        for col, rate in zip([c for c in df_warm.columns if c != "time_from_start"], null_rates):
            print(f"  {col}: {rate:.2%}")
            if ("lag" in col or "rolling" in col) and rate > 0.05:
                logger.warning(f"Feature {col} has >5% null rate ({rate:.2%}) after 168h warm-up!")
    else:
        print("  Not enough data to assess 168h warm-up.")
        
    print("="*40 + "\n")

    # Fill missing congestion with 0
    df = df.fill_null(0.0)

    # 3. Materialize
    out_path = out_dir / "features_latest.parquet"
    df.write_parquet(out_path)
    logger.info("Features materialized to %s", out_path)


if __name__ == "__main__":
    main()
