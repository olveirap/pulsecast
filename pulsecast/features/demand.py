"""
demand.py – Feature engineering from hourly TLC pickup counts.

Generates the following features for each (route_id / PULocationID, hour):
  - lag_1h, lag_2h, lag_3h, lag_24h, lag_168h  (direct lags)
  - rolling_mean_3h, rolling_mean_24h, rolling_mean_168h
  - ewm_trend_24h   (exponentially-weighted mean, span=24)
  - yoy_ratio       (volume / same-hour-last-year, clipped to [0.01, 100])
"""

from __future__ import annotations

import polars as pl


def _lag(col: str, hours: int) -> pl.Expr:
    return pl.col(col).shift(hours).alias(f"lag_{hours}h")


def _rolling_mean(col: str, window: int) -> pl.Expr:
    return (
        pl.col(col)
        .shift(1)
        .rolling_mean(window_size=window, min_periods=1)
        .alias(f"rolling_mean_{window}h")
    )


def build_demand_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: ``route_id`` (int), ``hour`` (Datetime),
        ``volume`` (int/float).  Rows must be sorted by (route_id, hour).

    Returns
    -------
    pl.DataFrame
        Original columns plus engineered demand features.
    """
    df = df.sort(["route_id", "hour"])

    df = df.with_columns(
        [
            # --- direct lags ---
            pl.col("volume").shift(1).over("route_id").alias("lag_1h"),
            pl.col("volume").shift(2).over("route_id").alias("lag_2h"),
            pl.col("volume").shift(3).over("route_id").alias("lag_3h"),
            pl.col("volume").shift(24).over("route_id").alias("lag_24h"),
            pl.col("volume").shift(168).over("route_id").alias("lag_168h"),
            # --- rolling means (excluding current row via shift(1)) ---
            pl.col("volume")
            .shift(1)
            .rolling_mean(window_size=3, min_periods=1)
            .over("route_id")
            .alias("rolling_mean_3h"),
            pl.col("volume")
            .shift(1)
            .rolling_mean(window_size=24, min_periods=1)
            .over("route_id")
            .alias("rolling_mean_24h"),
            pl.col("volume")
            .shift(1)
            .rolling_mean(window_size=168, min_periods=1)
            .over("route_id")
            .alias("rolling_mean_168h"),
            # --- exponentially-weighted mean (span=24) ---
            pl.col("volume")
            .shift(1)
            .ewm_mean(span=24, min_periods=1, adjust=False)
            .over("route_id")
            .alias("ewm_trend_24h"),
        ]
    )

    # --- year-over-year ratio (volume / volume 8760 h ago) ---
    df = df.with_columns(
        pl.col("volume").shift(8760).over("route_id").alias("_vol_yoy")
    ).with_columns(
        (pl.col("volume") / pl.col("_vol_yoy").clip(lower_bound=1e-3))
        .clip(lower_bound=0.01, upper_bound=100.0)
        .alias("yoy_ratio")
    ).drop("_vol_yoy")

    return df
