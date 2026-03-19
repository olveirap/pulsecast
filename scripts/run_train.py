"""
run_train.py – Model training pipeline for Pulsecast.

Loads engineered features, trains baseline, LightGBM, and TFT models,
and logs metrics to MLflow. Saves model artifacts for inference.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from pulsecast.models.baseline import BaselineForecaster
from pulsecast.models.lgbm import LGBMForecaster
from pulsecast.models.tft import TFTForecaster
from scripts.pipeline_config import LGBM_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame]:
    """Split features into training/validation and target."""
    logger.info("Preparing data: dropping nulls and sorting...")
    # Drop rows with NaNs from lags/rolling windows
    df = df.drop_nulls()
    
    missing_features = sorted(set(LGBM_FEATURES) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    # Sort for time-based split
    df = df.sort("hour")

    # Split on an hour cutoff so rows from the same timestamp stay together.
    unique_hours = df.select("hour").unique().sort("hour")
    if unique_hours.height < 2:
        raise ValueError("Need at least two distinct timestamps for train/validation split")
    cutoff_index = max(1, int(unique_hours.height * 0.8))
    if cutoff_index >= unique_hours.height:
        cutoff_index = unique_hours.height - 1
    cutoff_ts = unique_hours.row(cutoff_index)[0]

    train_df = df.filter(pl.col("hour") < cutoff_ts)
    val_df = df.filter(pl.col("hour") > cutoff_ts)
    dropped_at_cutoff = df.filter(pl.col("hour") == cutoff_ts).height
    if dropped_at_cutoff:
        logger.info(
            "Dropping %d rows at split boundary hour (%s) to prevent leakage.",
            dropped_at_cutoff,
            cutoff_ts,
        )
    if train_df.is_empty() or val_df.is_empty():
        raise ValueError("Time-based split produced an empty train or validation set")

    X_train = train_df.select(LGBM_FEATURES).to_numpy()
    y_train = train_df.select("volume").to_numpy().flatten()
    X_val = val_df.select(LGBM_FEATURES).to_numpy()
    y_val = val_df.select("volume").to_numpy().flatten()
    
    logger.info("Data prepared: %d train rows, %d val rows.", len(train_df), len(val_df))
    return X_train, y_train, X_val, y_val, train_df, val_df


def train_baseline(train_df: pl.DataFrame, models_dir: Path) -> None:
    logger.info("Starting baseline training...")
    with mlflow.start_run(run_name="Baseline-MSTL", nested=True):
        forecaster = BaselineForecaster()
        forecaster.fit(train_df)
        
        # Save baseline
        path = models_dir / "baseline.pkl"
        with open(path, "wb") as f:
            pickle.dump(forecaster, f)
        mlflow.log_artifact(str(path))
        logger.info("Baseline saved to %s", path)


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, models_dir: Path) -> LGBMForecaster:
    logger.info("Starting LightGBM training...")
    with mlflow.start_run(run_name="LightGBM-Quantile", nested=True):
        forecaster = LGBMForecaster()
        forecaster.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        logger.info("Running LightGBM cross-validation...")
        # Use a dedicated instance so CV retraining does not overwrite persisted models.
        cv_forecaster = LGBMForecaster()
        cv_results = cv_forecaster.cross_validate(X_train, y_train, n_splits=3)
        for fold, res in enumerate(cv_results):
            for q_name, loss in res.items():
                mlflow.log_metric(f"fold{fold}_{q_name}_pinball", loss)
        
        # Save model
        path = models_dir / "lgbm_forecaster.pkl"
        with open(path, "wb") as f:
            pickle.dump(forecaster, f)
        mlflow.log_artifact(str(path))
        logger.info("LightGBM saved to %s", path)
        return forecaster


def train_tft(train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
    logger.info("Starting TFT training...")
    # TFT expects a time_idx
    train_df = train_df.with_columns(
        (pl.col("hour").rank("dense") - 1).cast(pl.Int32).alias("time_idx")
    )
    # Correctly aligning time_idx for validation
    max_time_idx = train_df.select(pl.col("time_idx").max().cast(pl.Int64)).item()
    offset = int(max_time_idx) + 1
    val_df = val_df.with_columns(
        (pl.col("hour").rank("dense") - 1 + offset).cast(pl.Int32).alias("time_idx")
    )

    with mlflow.start_run(run_name="TFT", nested=True):
        forecaster = TFTForecaster(max_epochs=5)  # low epochs for demo
        forecaster.fit(train_df.to_pandas(), val_df.to_pandas())
        
        # Log artifacts (checkpoints)
        mlflow.log_artifacts(str(forecaster.checkpoint_dir))
        logger.info("TFT training complete and checkpoints logged.")


def main() -> None:
    logger.info("Starting training pipeline...")
    feat_path = Path(os.getenv("FEATURES_DIR", "data/features")) / "features_latest.parquet"
    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    logger.info("Configuring MLflow tracking URI: %s", tracking_uri)
    models_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info("Setting MLflow experiment...")
    mlflow.set_experiment("Pulsecast-Demand-Forecasting")

    if not feat_path.exists():
        logger.error("Features not found at %s. Run 'make features' first.", feat_path)
        raise SystemExit(1)

    logger.info("Loading features from %s", feat_path)
    df = pl.read_parquet(feat_path)
    
    X_train, y_train, X_val, y_val, train_df, val_df = prepare_data(df)
    
    logger.info("Opening main MLflow run...")
    with mlflow.start_run(run_name="Pulsecast-Train-Pipeline"):
        train_baseline(train_df, models_dir)
        train_lgbm(X_train, y_train, X_val, y_val, models_dir)
        train_tft(train_df, val_df)
    logger.info("Training pipeline finished.")


if __name__ == "__main__":
    main()
