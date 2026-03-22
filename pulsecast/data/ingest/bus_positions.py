"""
bus_positions.py - Ingest GTFS-RT bus positions from S3 and compute travel time variance.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from datetime import UTC, date
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import psycopg2
import requests
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
_DSN = os.getenv("TIMESCALE_DSN")
_TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
_BUCKET_NAME = os.getenv("BUS_POSITIONS_BUCKET", "nycbuspositions")


def _safe_extract_zip(zf: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for member in zf.infolist():
        target_path = (destination / member.filename).resolve()
        if os.path.commonpath([str(destination), str(target_path)]) != str(destination):
            raise ValueError(f"Unsafe zip member path detected: {member.filename}")
    zf.extractall(destination)


def _get_taxi_zones() -> gpd.GeoDataFrame:
    """Download and load TLC taxi zones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        taxi_zip = tmp_root / "taxi_zones.zip"
        
        logger.info("Downloading TLC taxi zones...")
        response = requests.get(_TAXI_ZONES_URL, timeout=120)
        response.raise_for_status()
        taxi_zip.write_bytes(response.content)
        
        with zipfile.ZipFile(taxi_zip) as zf:
            shapefiles = [name for name in zf.namelist() if name.lower().endswith(".shp")]
            if not shapefiles:
                raise ValueError("Taxi zones zip has no .shp file")
            shp_name = shapefiles[0]
            extract_dir = tmp_root / "taxi_zones_extracted"
            _safe_extract_zip(zf, extract_dir)
        
        zones = gpd.read_file(extract_dir / shp_name)
        
    if "LocationID" not in zones.columns:
        raise ValueError("Taxi zones dataset is missing required LocationID column")
    
    if zones.crs is None:
        zones = zones.set_crs("EPSG:4326")
    else:
        zones = zones.to_crs("EPSG:4326")
        
    return zones[["LocationID", "geometry"]].rename(columns={"LocationID": "zone_id"})


def fetch_bus_positions(target_date: date) -> pd.DataFrame:
    """Fetch bus positions from public S3 for the target date."""
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    key = f"{target_date.year}/{target_date.month:02d}/{target_date.year}-{target_date.month:02d}-{target_date.day:02d}-bus-positions.csv.xz"
    
    logger.info("Fetching s3://%s/%s", _BUCKET_NAME, key)
    try:
        response = s3_client.get_object(Bucket=_BUCKET_NAME, Key=key)
        df = pd.read_csv(response["Body"], compression="xz")
        return df
    except Exception as e:
        logger.error("Failed to fetch %s: %s", key, e)
        return pd.DataFrame()


def compute_variance(df: pd.DataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatially join and compute travel time variance per zone and hour."""
    if df.empty:
        return pd.DataFrame(columns=["zone_id", "hour", "travel_time_var", "sample_count"])
        
    if "segment_travel_time" not in df.columns:
        if "travel_time" in df.columns:
            df["segment_travel_time"] = df["travel_time"]
        else:
            logger.warning("Missing segment_travel_time column. Using 0.0 as fallback.")
            df["segment_travel_time"] = 0.0

    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    elif "time" in df.columns:
        df["hour"] = pd.to_datetime(df["time"]).dt.floor("h")
    else:
        df["hour"] = pd.Timestamp("today").floor("h")

    if "latitude" in df.columns and "longitude" in df.columns:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )
    else:
        logger.error("Missing latitude/longitude columns")
        return pd.DataFrame()

    joined = gpd.sjoin(gdf, zones, how="inner", predicate="intersects")
    
    agg = joined.groupby(["zone_id", "hour"]).agg(
        travel_time_var=("segment_travel_time", "var"),
        sample_count=("segment_travel_time", "count")
    ).reset_index()
    
    agg["travel_time_var"] = agg["travel_time_var"].fillna(0.0)
    
    return agg


def write_to_db(df: pd.DataFrame, dsn: str | None) -> int:
    """Upsert aggregated data into the congestion table."""
    if df.empty or dsn is None:
        return 0

    conn = psycopg2.connect(dsn)
    try:
        params = [
            (
                int(row["zone_id"]),
                row["hour"].to_pydatetime().replace(tzinfo=UTC) if row["hour"].tzinfo is None else row["hour"].to_pydatetime(),
                float(row["travel_time_var"]),
                int(row["sample_count"]),
                False  # disruption_flag
            )
            for _, row in df.iterrows()
        ]

        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO congestion (zone_id, hour, travel_time_var, sample_count, disruption_flag)
                    VALUES %s
                    ON CONFLICT (zone_id, hour)
                    DO UPDATE SET 
                        travel_time_var = EXCLUDED.travel_time_var,
                        sample_count = EXCLUDED.sample_count,
                        disruption_flag = EXCLUDED.disruption_flag;
                    """,
                    params,
                )
        return len(params)
    finally:
        conn.close()


def process_date(target_date: date, zones: gpd.GeoDataFrame, dsn: str | None = _DSN) -> None:
    """Process a single day of bus positions."""
    df = fetch_bus_positions(target_date)
    if df.empty:
        logger.warning("No data to process for %s", target_date)
        return
        
    agg = compute_variance(df, zones)
    if agg.empty:
        logger.warning("No matched records after spatial join for %s", target_date)
        return
        
    written = write_to_db(agg, dsn)
    logger.info("Processed and upserted %d records for %s", written, target_date)


def main():
    target_date = date.today() - pd.Timedelta(days=1)
    zones = _get_taxi_zones()
    process_date(target_date, zones)


if __name__ == "__main__":
    main()
