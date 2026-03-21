"""
nyc_events.py – Fetch event data from NYC Open Data.
"""

import logging
from datetime import date

import requests

logger = logging.getLogger(__name__)


def fetch_nyc_events() -> set[date]:
    """
    Fetch major NYC events from the Open Data API.
    Returns a set of date objects.
    """
    url = "https://data.cityofnewyork.us/resource/byf3-74rd.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list):
            logger.warning(f"Unexpected API response format: {type(data)}")
            return set()

        event_dates = set()
        for item in data:
            # Standard field for many NYC event datasets
            start_date_str = item.get("event_start_date")
            if start_date_str:
                try:
                    # API dates often look like '2024-03-17T00:00:00.000'
                    # We extract the date portion.
                    d_str = start_date_str.split("T")[0]
                    event_dates.add(date.fromisoformat(d_str))
                except (ValueError, IndexError):
                    continue
        return event_dates
    except Exception as e:
        logger.warning(f"Failed to fetch NYC events from API: {e}")
        return set()
