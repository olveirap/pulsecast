import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl

from scripts.update_disruption_flag import main


class TestUpdateDisruptionFlag(unittest.TestCase):
    @patch("scripts.update_disruption_flag.psycopg2.connect")
    @patch("scripts.update_disruption_flag.pl.read_database")
    @patch("scripts.update_disruption_flag.execute_values")
    @patch("scripts.update_disruption_flag.build_congestion_features")
    def test_main_updates_mismatched_flags(self, mock_build_features, mock_execute_values, mock_read_database, mock_connect):
        # Setup mock data
        mock_data = pl.DataFrame({
            "zone_id": [1, 1],
            "hour": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 1, 0)
            ],
            "travel_time_var": [100.0, 1.0],
            "sample_count": [15, 15],
            "disruption_flag": [False, False]
        }).with_columns(pl.col("hour").cast(pl.Datetime))
        mock_read_database.return_value = mock_data
        
        # Mock features output: Row 1 has a disruption (1), Row 2 does not (0)
        mock_features = mock_data.rename({"disruption_flag": "old_flag"}).with_columns(
            pl.Series("disruption_flag", [1, 0], dtype=pl.Int8)
        )
        mock_build_features.return_value = mock_features
        
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        
        # Run main
        main()
        
        # Verify execute_values was called with the one update (Row 1 changed from False to 1)
        self.assertTrue(mock_execute_values.called)
        args, kwargs = mock_execute_values.call_args
        # args[2] is the rows list
        self.assertEqual(len(args[2]), 1)
        self.assertEqual(args[2][0][2], 1) # new_flag

if __name__ == "__main__":
    unittest.main()
