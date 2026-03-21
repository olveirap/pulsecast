"""
bus_positions_backfill.py - Stub for historical GTFS-RT bus positions ingestion.
"""

import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="Start date")
    parser.add_argument("--end", type=str, help="End date")
    args = parser.parse_args()
    logger.info(f"Stub: bus_positions_backfill ran successfully for {args.start} to {args.end}.")

if __name__ == "__main__":
    main()
