"""
tft.py – Temporal Fusion Transformer via pytorch-forecasting.

Trains a TFT model for 7-day (168-hour) probabilistic demand forecasting.

Usage
-----
>>> from models.tft import TFTForecaster
>>> forecaster = TFTForecaster()
>>> forecaster.fit(train_df)
>>> predictions = forecaster.predict(val_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
except ImportError as exc:
    raise ImportError(
        "pytorch-forecasting is required. "
        "Install it with: pip install pytorch-forecasting pytorch-lightning"
    ) from exc


_HORIZON = 168           # 7 days × 24 h
_ENCODER_LENGTH = 336    # 14-day look-back
_QUANTILES = [0.1, 0.5, 0.9]


class TFTForecaster:
    """
    Wraps pytorch-forecasting's TemporalFusionTransformer for hourly demand.

    Parameters
    ----------
    max_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Initial learning rate.
    hidden_size : int
        TFT hidden state dimension.
    checkpoint_dir : str | Path
        Directory for model checkpoints.
    """

    def __init__(
        self,
        max_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 3e-3,
        hidden_size: int = 64,
        checkpoint_dir: str | Path = "checkpoints/tft",
    ) -> None:
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model: TemporalFusionTransformer | None = None
        self._dataset: TimeSeriesDataSet | None = None

    def _make_dataset(self, df: pd.DataFrame, predict: bool = False) -> TimeSeriesDataSet:
        """Build a TimeSeriesDataSet from a pandas DataFrame."""
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="volume",
            group_ids=["route_id"],
            min_encoder_length=_ENCODER_LENGTH // 2,
            max_encoder_length=_ENCODER_LENGTH,
            min_prediction_length=1,
            max_prediction_length=_HORIZON,
            static_categoricals=["route_id"],
            time_varying_known_reals=[
                "time_idx",
                "hour_of_day",
                "dow",
                "week_of_year",
                "days_to_next_us_holiday",
                "nyc_event_flag",
            ],
            time_varying_unknown_reals=[
                "volume",
                "delay_index",
                "disruption_flag",
            ],
            target_normalizer=GroupNormalizer(
                groups=["route_id"], transformation="softplus"
            ),
            allow_missing_timesteps=True,
            predict_mode=predict,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "TFTForecaster":
        """
        Train the TFT model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain columns: route_id, time_idx (int), volume,
            hour_of_day, dow, week_of_year, days_to_next_us_holiday,
            nyc_event_flag, delay_index, disruption_flag.
        val_df : pd.DataFrame | None
            Optional validation set.
        """
        self._dataset = self._make_dataset(train_df)
        train_loader = self._dataset.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=4
        )

        val_loader = None
        if val_df is not None:
            val_dataset = TimeSeriesDataSet.from_dataset(self._dataset, val_df, predict=True)
            val_loader = val_dataset.to_dataloader(
                train=False, batch_size=self.batch_size * 4, num_workers=4
            )

        self._model = TemporalFusionTransformer.from_dataset(
            self._dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=QuantileLoss(quantiles=_QUANTILES),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        callbacks: list[Any] = [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
            ),
        ]

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=callbacks,
            enable_progress_bar=True,
        )
        trainer.fit(
            self._model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        logger.info("TFTForecaster training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, list[float]]:
        """
        Generate probabilistic forecasts for each group in *df*.

        Returns
        -------
        dict with keys ``"p10"``, ``"p50"``, ``"p90"``.
        """
        if self._model is None or self._dataset is None:
            raise RuntimeError("Call fit() before predict().")
        pred_dataset = TimeSeriesDataSet.from_dataset(self._dataset, df, predict=True)
        pred_loader = pred_dataset.to_dataloader(
            train=False, batch_size=self.batch_size * 4, num_workers=4
        )
        raw = self._model.predict(pred_loader, mode="quantiles", return_index=True)
        predictions = raw[0]
        return {
            "p10": predictions[:, :, 0].flatten().tolist(),
            "p50": predictions[:, :, 1].flatten().tolist(),
            "p90": predictions[:, :, 2].flatten().tolist(),
        }
