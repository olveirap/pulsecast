from datetime import date
from unittest.mock import patch

from pulsecast.data.ingest.bus_positions_backfill import backfill


@patch("pulsecast.data.ingest.bus_positions_backfill._get_taxi_zones")
@patch("pulsecast.data.ingest.bus_positions_backfill.get_processed_days")
@patch("pulsecast.data.ingest.bus_positions_backfill.process_date")
def test_backfill_loop(mock_process, mock_get_processed, mock_get_zones):
    # Setup mocks
    mock_get_processed.return_value = {date(2024, 1, 2)}  # Jan 2 is already processed
    mock_get_zones.return_value = "mock_zones"
    
    start = date(2024, 1, 1)
    end = date(2024, 1, 3)
    
    backfill(start, end, dsn=None)
    
    # Assertions
    # It should skip Jan 2, and process Jan 1 and Jan 3.
    assert mock_process.call_count == 2
    mock_process.assert_any_call(date(2024, 1, 1), "mock_zones", None)
    mock_process.assert_any_call(date(2024, 1, 3), "mock_zones", None)
