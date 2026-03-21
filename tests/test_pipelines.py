"""
tests/test_pipelines.py – Integration tests for the pipeline scripts.

Mocks external dependencies (TimescaleDB, MLflow) and model training to
verify that the scripts run without errors and perform correct I/O.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

from scripts.pipeline_config import LGBM_FEATURES

# ---------------------------------------------------------------------------
# Mocks for heavy imports
# ---------------------------------------------------------------------------

# Mocking these before any script imports to avoid loading Torch/StatsForecast
mock_sf = MagicMock()
mock_pl = MagicMock()
mock_pf = MagicMock()

sys.modules["statsforecast"] = mock_sf
sys.modules["statsforecast.models"] = MagicMock()
sys.modules["pytorch_lightning"] = mock_pl
sys.modules["pytorch_lightning.callbacks"] = MagicMock()
sys.modules["pytorch_forecasting"] = mock_pf
sys.modules["pytorch_forecasting.data"] = MagicMock()
sys.modules["pytorch_forecasting.metrics"] = MagicMock()

# ---------------------------------------------------------------------------
# Picklable dummies for tests
# ---------------------------------------------------------------------------

class DummyForecaster:
    """Minimal picklable class for testing run_train and run_export."""
    def __init__(self):
        self.checkpoint_dir = "dummy_checkpoints"
    def fit(self, *args, **kwargs): return self
    def predict(self, *args, **kwargs): return {"p10": [0], "p50": [0], "p90": [0]}
    def cross_validate(self, *args, **kwargs): return [{"p10": 0.1, "p50": 0.05, "p90": 0.1}]

# ---------------------------------------------------------------------------
# run_features.py
# ---------------------------------------------------------------------------

@patch("psycopg2.connect")
@patch("polars.read_database")
def test_run_features(mock_read_db, mock_connect):
    """run_features should load data, build features, and write parquet."""
    print("\nStarting test_run_features...")
    from scripts.run_features import main

    # Mock DB data for demand
    mock_df_demand = pl.DataFrame({
        "route_id": [132, 132],
        "hour": [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1)],
        "volume": [10, 20],
        "avg_duration": [300.0, 310.0],
    }, schema={"route_id": pl.Int32, "hour": pl.Datetime, "volume": pl.Int32, "avg_duration": pl.Float64})

    # Mock DB data for routes
    mock_df_routes = pl.DataFrame({
        "route_id": [132],
        "origin_zone_id": [10],
        "destination_zone_id": [20],
    }, schema={"route_id": pl.Int32, "origin_zone_id": pl.Int32, "destination_zone_id": pl.Int32})

    # Mock DB data for congestion
    mock_df_congestion = pl.DataFrame({
        "zone_id": [10, 20],
        "hour": [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 0)],
        "travel_time_var": [0.5, 0.6],
        "sample_count": [15, 15],
    }, schema={"zone_id": pl.Int32, "hour": pl.Datetime, "travel_time_var": pl.Float64, "sample_count": pl.Int32})

    mock_read_db.side_effect = [mock_df_demand, mock_df_routes, mock_df_congestion]
    print("Mock data prepared.")    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"FEATURES_DIR": tmpdir}):
            print("Calling run_features.main()...")
            main()
            print("run_features.main() completed.")
            
            out_path = Path(tmpdir) / "features_latest.parquet"
            assert out_path.exists()
            
            df_out = pl.read_parquet(out_path)
            missing = sorted(set(LGBM_FEATURES) - set(df_out.columns))
            assert not missing, f"run_features output missing required train columns: {missing}"


# ---------------------------------------------------------------------------
# run_train.py
# ---------------------------------------------------------------------------

@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_metric")
@patch("mlflow.log_artifact")
@patch("mlflow.log_artifacts")
@patch("scripts.run_train.BaselineForecaster")
@patch("scripts.run_train.LGBMForecaster")
@patch("scripts.run_train.TFTForecaster")
def test_run_train(mock_tft, mock_lgbm, mock_baseline, mock_log_artifacts, mock_log_artifact, mock_log_metric, mock_start_run, mock_set_experiment, mock_set_uri):
    """run_train should load features and train models with MLflow logging."""
    print("\nStarting test_run_train...")
    from scripts.run_train import main
    
    # Configure mocks with picklable return values
    mock_baseline.return_value = DummyForecaster()
    mock_lgbm.return_value = DummyForecaster()
    mock_tft.return_value = DummyForecaster()
    print("Mocks configured with picklable dummies.")
    
    # Create dummy features
    with tempfile.TemporaryDirectory() as tmpdir:
        feat_dir = Path(tmpdir) / "features"
        feat_dir.mkdir()
        feat_path = feat_dir / "features_latest.parquet"
        
        n_rows = 200 # enough for splits
        rng = np.random.default_rng(42)
        
        start_dt = datetime(2024, 1, 1)
        end_dt = start_dt + timedelta(hours=n_rows - 1)
        
        df = pl.DataFrame({
            "route_id": [132] * n_rows,
            "hour": pl.datetime_range(start_dt, end_dt, "1h", eager=True),
            "volume": rng.integers(0, 100, n_rows),
            "avg_duration": rng.uniform(100, 500, n_rows),
            "origin_travel_time_var": rng.uniform(0, 1, n_rows),
            "dest_travel_time_var": rng.uniform(0, 1, n_rows),
            "hour_of_day": [i % 24 for i in range(n_rows)],
            "dow": [(i // 24) % 7 for i in range(n_rows)],
            "month": [1] * n_rows,
            "week_of_year": [1] * n_rows,
            "is_weekend": [0] * n_rows,
            "days_to_next_us_holiday": [5] * n_rows,
            "nyc_event_flag": [0] * n_rows,
            "lag_1h": rng.uniform(0, 100, n_rows),
            "lag_2h": rng.uniform(0, 100, n_rows),
            "lag_3h": rng.uniform(0, 100, n_rows),
            "lag_24h": rng.uniform(0, 100, n_rows),
            "lag_168h": rng.uniform(0, 100, n_rows),
            "rolling_mean_3h": rng.uniform(0, 100, n_rows),
            "rolling_mean_24h": rng.uniform(0, 100, n_rows),
            "rolling_mean_168h": rng.uniform(0, 100, n_rows),
            "ewm_trend_24h": rng.uniform(0, 100, n_rows),
            "yoy_ratio": rng.uniform(0.5, 1.5, n_rows),
            "origin_delay_index_lag1": rng.uniform(0, 1, n_rows),
            "origin_delay_index_rolling3h": rng.uniform(0, 1, n_rows),
            "origin_disruption_flag": [0] * n_rows,
            "dest_delay_index_lag1": rng.uniform(0, 1, n_rows),
            "dest_delay_index_rolling3h": rng.uniform(0, 1, n_rows),
            "dest_disruption_flag": [0] * n_rows,
            "duration_lag_1h": rng.uniform(100, 500, n_rows),
            "duration_lag_24h": rng.uniform(100, 500, n_rows),
            "duration_lag_168h": rng.uniform(100, 500, n_rows),
            "duration_rolling_mean_3h": rng.uniform(100, 500, n_rows),
            "duration_rolling_mean_24h": rng.uniform(100, 500, n_rows),
        })
        missing = sorted(set(LGBM_FEATURES) - set(df.columns))
        assert not missing, f"Synthetic training data missing required columns: {missing}"
        df.write_parquet(feat_path)
        print(f"Dummy features written to {feat_path}.")
        
        models_dir = Path(tmpdir) / "models"
        
        with patch.dict(os.environ, {
            "FEATURES_DIR": str(feat_dir),
            "MODELS_DIR": str(models_dir),
            "MLFLOW_TRACKING_URI": "http://localhost:5000"
        }):
            print("Calling run_train.main()...")
            main()
            print("run_train.main() completed.")
            
            assert (models_dir / "baseline.pkl").exists()
            assert (models_dir / "lgbm_forecaster.pkl").exists()
            assert mock_start_run.called


def test_prepare_data_drops_split_boundary_hour():
    """prepare_data should exclude all rows at the cutoff hour from both splits."""
    from scripts.run_train import prepare_data

    n_hours = 10
    route_ids = [101, 202]
    start_dt = datetime(2024, 1, 1)

    hours: list[datetime] = []
    routes: list[int] = []
    for h in range(n_hours):
        for rid in route_ids:
            hours.append(start_dt + timedelta(hours=h))
            routes.append(rid)

    n_rows = len(hours)
    df = pl.DataFrame({
        "route_id": routes,
        "hour": hours,
        "volume": [10.0] * n_rows,
        "avg_duration": [300.0] * n_rows,
        "origin_travel_time_var": [0.1] * n_rows,
        "dest_travel_time_var": [0.1] * n_rows,
        "hour_of_day": [h.hour for h in hours],
        "dow": [1] * n_rows,
        "month": [1] * n_rows,
        "week_of_year": [1] * n_rows,
        "is_weekend": [0] * n_rows,
        "days_to_next_us_holiday": [5] * n_rows,
        "nyc_event_flag": [0] * n_rows,
        "lag_1h": [1.0] * n_rows,
        "lag_2h": [1.0] * n_rows,
        "lag_3h": [1.0] * n_rows,
        "lag_24h": [1.0] * n_rows,
        "lag_168h": [1.0] * n_rows,
        "rolling_mean_3h": [1.0] * n_rows,
        "rolling_mean_24h": [1.0] * n_rows,
        "rolling_mean_168h": [1.0] * n_rows,
        "ewm_trend_24h": [1.0] * n_rows,
        "yoy_ratio": [1.0] * n_rows,
        "origin_delay_index_lag1": [0.1] * n_rows,
        "origin_delay_index_rolling3h": [0.1] * n_rows,
        "origin_disruption_flag": [0] * n_rows,
        "dest_delay_index_lag1": [0.1] * n_rows,
        "dest_delay_index_rolling3h": [0.1] * n_rows,
        "dest_disruption_flag": [0] * n_rows,
        "duration_lag_1h": [300.0] * n_rows,
        "duration_lag_24h": [300.0] * n_rows,
        "duration_lag_168h": [300.0] * n_rows,
        "duration_rolling_mean_3h": [300.0] * n_rows,
        "duration_rolling_mean_24h": [300.0] * n_rows,
    })

    _, _, _, _, train_df, val_df = prepare_data(df)

    cutoff_ts = start_dt + timedelta(hours=8)
    dropped_expected = len(route_ids)
    assert train_df.filter(pl.col("hour") == cutoff_ts).is_empty()
    assert val_df.filter(pl.col("hour") == cutoff_ts).is_empty()
    assert train_df.height + val_df.height == df.height - dropped_expected


# ---------------------------------------------------------------------------
# run_export.py
# ---------------------------------------------------------------------------

@patch("pulsecast.models.export.export_lgbm_to_onnx")
def test_run_export(mock_export):
    """run_export should load pickle and call export function."""
    print("\nStarting test_run_export...")
    from scripts.run_export import main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir)
        pickle_path = models_dir / "lgbm_forecaster.pkl"
        
        forecaster = DummyForecaster()
        with open(pickle_path, "wb") as f:
            import pickle
            pickle.dump(forecaster, f)
        print(f"Dummy forecaster pickled at {pickle_path}.")
            
        mock_export.return_value = {"p10": Path("lgbm_p10.onnx")}
        
        with patch.dict(os.environ, {"MODELS_DIR": str(models_dir)}):
            print("Calling run_export.main()...")
            main()
            print("run_export.main() completed.")
            assert mock_export.called
