from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from scripts.update_disruption_flag import main


@pytest.fixture
def mock_db_data():
    """Create mock database data for testing."""
    return pl.DataFrame({
        "zone_id": [1, 1],
        "hour": [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 1, 0)
        ],
        "travel_time_var": [100.0, 1.0],
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


@patch("scripts.update_disruption_flag.execute_values")
@patch("scripts.update_disruption_flag.build_congestion_features")
def test_main_updates_mismatched_flags(
    mock_build_features,
    mock_execute_values,
    mock_connect_fixtures,
    mock_db_data
):
    """Test that main() updates only zones where disruption flags mismatch."""
    # Arrange
    mock_connect, mock_read_db, mock_conn, mock_cur = mock_connect_fixtures
    
    # Mock features output: Row 1 has a disruption (1), Row 2 does not (0)
    mock_features = mock_db_data.rename({"disruption_flag": "old_flag"}).with_columns(
        pl.Series("disruption_flag", [1, 0], dtype=pl.Int8)
    )
    mock_build_features.return_value = mock_features
    
    # Act
    main()
    
    # Assert
    assert mock_execute_values.called, "execute_values should be called to update mismatched flags"
    args, kwargs = mock_execute_values.call_args
    # args[2] is the rows list
    assert len(args[2]) == 1, "Only one row should be updated (the one with mismatched flag)"
    assert args[2][0][2] == 1, "New flag value should be 1 (disruption detected)"
