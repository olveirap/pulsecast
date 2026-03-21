"""
update_disruption_flag.py – Post-processing job to compute and update disruption_flag.

Computes the binary disruption_flag (travel_time_var > µ + 2σ over trailing 168h)
for all rows in the congestion table and updates the database.

Memory-efficient approach: processes data in chunks per zone_id to avoid loading
the entire congestion table into memory at once.
"""

import logging
import os
from typing import Any

import polars as pl
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from pulsecast.features.congestion import build_congestion_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast",
)


def fetch_zone_ids(conn: Any) -> list[int]:
    """Fetch all unique zone_ids from the congestion table."""
    query = "SELECT DISTINCT zone_id FROM congestion ORDER BY zone_id"
    df = pl.read_database(query, conn)
    return df["zone_id"].to_list()


def fetch_zone_data(conn: Any, zone_id: int) -> pl.DataFrame:
    """Fetch congestion data for a specific zone_id."""
    query = """
    SELECT zone_id, hour, travel_time_var, sample_count, disruption_flag
    FROM congestion
    WHERE zone_id = %s
    ORDER BY hour
    """
    with conn.cursor() as cur:
        cur.execute(query, (zone_id,))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=columns)


def process_zone_data(df: pl.DataFrame) -> pl.DataFrame:
    """Compute disruption flags for a single zone's data."""
    # Ensure 'hour' is pl.Datetime
    df = df.with_columns(pl.col("hour").cast(pl.Datetime))
    
    # Rename existing flag to avoid collision with build_congestion_features output
    df = df.rename({"disruption_flag": "old_flag"})
    df_features = build_congestion_features(df)
    
    # Identify mismatches (build_congestion_features returns 0/1 as Int8)
    df_updates = df_features.filter(
        pl.col("disruption_flag").cast(pl.Boolean) != pl.col("old_flag")
    ).select(["zone_id", "hour", "disruption_flag"])
    
    return df_updates


def bulk_update_flags(conn: Any, updates: list[tuple]) -> int:
    """Bulk update disruption flags for multiple rows. Returns number of rows updated."""
    if not updates:
        return 0
    
    with conn.cursor() as cur:
        update_query = """
        UPDATE congestion AS c
        SET disruption_flag = v.new_flag::boolean
        FROM (VALUES %s) AS v(zone_id, hour, new_flag)
        WHERE c.zone_id = v.zone_id AND c.hour = v.hour
        """
        execute_values(cur, update_query, updates)
        rows_updated = cur.rowcount
    
    return rows_updated


def main():
    logger.info("Connecting to TimescaleDB to update disruption flags.")
    try:
        with psycopg2.connect(_DB_DSN) as conn:
            # 1. Fetch all unique zone_ids (small memory footprint)
            logger.info("Fetching unique zone_ids...")
            zone_ids = fetch_zone_ids(conn)
            
            if not zone_ids:
                logger.info("No congestion data found. Skipping update.")
                return
            
            logger.info("Found %d unique zones to process.", len(zone_ids))
            
            # 2. Process each zone individually (memory-efficient)
            all_updates: list[tuple[int, object, int]] = []
            total_rows_processed = 0
            
            for i, zone_id in enumerate(zone_ids, 1):
                # Fetch data for this zone only
                zone_df = fetch_zone_data(conn, zone_id)
                
                if zone_df.is_empty():
                    continue
                
                total_rows_processed += len(zone_df)
                
                # Compute features for this zone
                zone_updates = process_zone_data(zone_df)
                
                # Collect updates (convert to native Python types for execute_values)
                all_updates.extend(zone_updates.rows())
                
                # Progress logging
                if i % 10 == 0 or i == len(zone_ids):
                    logger.info(
                        "Processed %d/%d zones (%d rows), %d updates so far",
                        i, len(zone_ids), total_rows_processed, len(all_updates)
                    )
            
            if not all_updates:
                logger.info("No disruption flag updates required.")
                return
            
            logger.info(
                "Total: %d rows processed across %d zones, %d updates to apply",
                total_rows_processed, len(zone_ids), len(all_updates)
            )
            
            # 3. Perform bulk update
            rows_updated = bulk_update_flags(conn, all_updates)
            conn.commit()
            
            logger.info(
                "Successfully updated %d disruption flags across %d zones.",
                rows_updated, len(zone_ids)
            )

    except (psycopg2.Error, pl.exceptions.PolarsError) as e:
        logger.error("Error updating disruption flags: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
