"""
baseline.py – MSTL decomposition + AutoARIMA per route via statsforecast.

Usage
-----
>>> from models.baseline import BaselineForecaster
>>> model = BaselineForecaster()
>>> model.fit(train_df)          # pl.DataFrame with route_id, hour, volume
>>> forecasts = model.predict(horizon=168)
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

try:
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoARIMA
except ImportError as exc:
    raise ImportError(
        "statsforecast is required for BaselineForecaster. "
        "Install it with: pip install statsforecast"
    ) from exc


class BaselineForecaster:
    """
    Fits one MSTL+AutoARIMA model per unique ``route_id`` in the training data.

    Parameters
    ----------
    season_lengths : list[int]
        Seasonal periods (hours) to pass to MSTL.  Defaults to [24, 168]
        (daily and weekly seasonality).
    freq : str
        Pandas-compatible frequency string for the time index (default ``'h'``).
    """

    def __init__(
        self,
        season_lengths: list[int] | None = None,
        freq: str = "h",
    ) -> None:
        self.season_lengths = season_lengths or [24, 168]
        self.freq = freq
        self._sf: StatsForecast | None = None

    def fit(self, df: pl.DataFrame) -> "BaselineForecaster":
        """
        Fit the model.

        Parameters
        ----------
        df : pl.DataFrame
            Columns: ``route_id`` (int), ``hour`` (Datetime[us]), ``volume`` (numeric).
        """
        sf_df = (
            df.rename({"route_id": "unique_id", "hour": "ds", "volume": "y"})
            .select(["unique_id", "ds", "y"])
            .to_pandas()
        )
        sf_df["ds"] = sf_df["ds"].dt.tz_localize(None)

        models: list[Any] = [
            MSTL(
                season_length=self.season_lengths,
                trend_forecaster=AutoARIMA(),
            )
        ]
        self._sf = StatsForecast(models=models, freq=self.freq, n_jobs=-1)
        self._sf.fit(sf_df)
        logger.info("BaselineForecaster fitted on %d routes.", df["route_id"].n_unique())
        return self

    def predict(self, horizon: int = 168) -> pl.DataFrame:
        """
        Generate point forecasts for *horizon* steps ahead.

        Returns a Polars DataFrame with columns:
        ``route_id``, ``hour``, ``forecast``.
        """
        if self._sf is None:
            raise RuntimeError("Call fit() before predict().")
        forecast_pd = self._sf.predict(h=horizon)
        result = pl.from_pandas(forecast_pd.reset_index()).rename(
            {"unique_id": "route_id", "ds": "hour", "MSTL": "forecast"}
        )
        return result
