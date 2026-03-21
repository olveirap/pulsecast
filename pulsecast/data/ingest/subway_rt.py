"""
subway_rt.py - Stub for real-time subway GTFS-RT ingestion.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Stub: subway_rt ingest ran successfully.")

if __name__ == "__main__":
    main()
