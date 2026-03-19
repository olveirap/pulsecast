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
    
    # Mock DB data - use schema to be 100% sure
    schema = {
        "route_id": pl.Int32,
        "hour": pl.Datetime,
        "volume": pl.Int32,
        "delay_index": pl.Float64
    }
    mock_df = pl.DataFrame({
        "route_id": [132, 132],
        "hour": [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1)],
        "volume": [10, 20],
        "delay_index": [0.5, 0.6]
    }, schema=schema)
    mock_read_db.return_value = mock_df
    print("Mock data prepared.")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"FEATURES_DIR": tmpdir}):
            print("Calling run_features.main()...")
            main()
            print("run_features.main() completed.")
            
            out_path = Path(tmpdir) / "features_latest.parquet"
            assert out_path.exists()
            
            df_out = pl.read_parquet(out_path)
            assert "lag_1h" in df_out.columns
            assert "hour_of_day" in df_out.columns
            assert "disruption_flag" in df_out.columns


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
            "delay_index": rng.uniform(0, 1, n_rows),
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
            "delay_index_lag1": rng.uniform(0, 1, n_rows),
            "delay_index_rolling3h": rng.uniform(0, 1, n_rows),
            "disruption_flag": [0] * n_rows
        })
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
