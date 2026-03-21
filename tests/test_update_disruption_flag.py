from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from scripts.update_disruption_flag import main


@pytest.fixture
def mock_db_data():
    """Create mock database data for testing."""
    return pl.DataFrame({
        "zone_id": [1, 2],
        "hour": [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 0, 0)
        ],
        "travel_time_var": [100.0, 50.0],
        "sample_count": [15, 15],
        "disruption_flag": [False, False]
    }).with_columns(pl.col("hour").cast(pl.Datetime))


@pytest.fixture
def mock_connect(mock_db_data):
    """Mock psycopg2.connect and set up database read."""
    with patch("scripts.update_disruption_flag.psycopg2.connect") as mock_connect:
        with patch("scripts.update_disruption_flag.pl.read_database") as mock_read_db:
            mock_read_db.return_value = mock_db_data
            mock_conn = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cur
            yield mock_connect, mock_read_db, mock_conn, mock_cur


@patch("scripts.update_disruption_flag.bulk_update_flags")
@patch("scripts.update_disruption_flag.fetch_zone_data")
@patch("scripts.update_disruption_flag.build_congestion_features")
def test_main_updates_mismatched_flags(
    mock_build_features: MagicMock,
    mock_fetch_zone_data: MagicMock,
    mock_bulk_update_flags: MagicMock,
    mock_db_data: pl.DataFrame,
    mock_connect: tuple
):
    """Test that main() updates only zones where disruption flags mismatch."""
    
    # Unpack the mock_connect fixture
    mock_connect_func, mock_read_db, mock_conn, _ = mock_connect
    
    # Mock features output: Zone 1 has a disruption (1), Zone 2 does not (0)
    # Each call returns features for one zone
    mock_build_features.side_effect = [
        # Zone 1: disruption detected (flag changes from False to 1)
        mock_db_data.filter(pl.col("zone_id") == 1).rename({"disruption_flag": "old_flag"}).with_columns(
            pl.Series("disruption_flag", [1], dtype=pl.Int8)
        ),
        # Zone 2: no disruption (flag stays 0, matches old_flag=False)
        mock_db_data.filter(pl.col("zone_id") == 2).rename({"disruption_flag": "old_flag"}).with_columns(
            pl.Series("disruption_flag", [0], dtype=pl.Int8)
        ),
    ]
    
    # Mock zone data for each zone
    mock_fetch_zone_data.side_effect = [
        mock_db_data.filter(pl.col("zone_id") == 1),
        mock_db_data.filter(pl.col("zone_id") == 2),
    ]
    
    # Mock bulk_update_flags to return number of rows updated
    mock_bulk_update_flags.return_value = 1
    
    # Act
    main()
    
    # Assert psycopg2.connect was called with correct DSN
    from scripts.update_disruption_flag import _DB_DSN
    mock_connect_func.assert_called_once_with(_DB_DSN)
    
    # Assert pl.read_database was called to fetch zone_ids
    mock_read_db.assert_called_once()
    
    # Assert fetch_zone_data was called twice (once per zone)
    assert mock_fetch_zone_data.call_count == 2
    
    # Assert build_congestion_features was called twice (once per zone)
    assert mock_build_features.call_count == 2
    
    # Assert bulk_update_flags was called with the mismatched row
    mock_bulk_update_flags.assert_called_once()
    args, _ = mock_bulk_update_flags.call_args
    updates = args[1]  # Second argument is the updates list
    assert len(updates) == 1, "Only one row should be updated (the one with mismatched flag)"
    assert updates[0][2] == 1, "New flag value should be 1 (disruption detected)"
    
    # Assert connection commit was called
    mock_conn.commit.assert_called_once()
