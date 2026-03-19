"""
congestion.py – Features derived from bus position travel time variance.

Features produced per (zone_id, hour):
  - delay_index_lag1      (travel_time_var at t-1 h)
  - delay_index_rolling3h (mean over previous 3 hours)
  - disruption_flag       (1 if travel_time_var > µ + 2σ over the trailing 168 h window)
  - low_confidence_flag   (1 if sample_count < 10)

NOTE: Output feature names 'delay_index_*' are preserved for backward compatibility
with downstream models, but they now refer to travel time variance.
"""

from __future__ import annotations

import polars as pl


def build_congestion_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: ``zone_id`` (int), ``hour`` (Datetime),
        ``travel_time_var`` (float), ``sample_count`` (int).
        Rows must be sorted by (zone_id, hour).

    Returns
    -------
    pl.DataFrame
        Original columns plus congestion feature columns.
    """
    if "travel_time_var" not in df.columns:
        if "delay_index" in df.columns:
            df = df.with_columns(pl.col("delay_index").alias("travel_time_var"))
        else:
            raise ValueError("build_congestion_features requires travel_time_var or delay_index")

    if "sample_count" not in df.columns:
        # Keep output schema stable for legacy inputs without sampling metadata.
        df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("sample_count"))

    df = df.sort(["zone_id", "hour"])

    df = df.with_columns(
        [
            # lag-1 travel_time_var (aliased to delay_index_lag1)
            pl.col("travel_time_var")
            .shift(1)
            .over("zone_id")
            .alias("delay_index_lag1"),
            # rolling 3-hour mean (excluding current observation)
            pl.col("travel_time_var")
            .shift(1)
            .rolling_mean(window_size=3)
            .over("zone_id")
            .alias("delay_index_rolling3h"),
            # low_confidence_flag: 1 if sample_count < 10
            (pl.col("sample_count") < 10).cast(pl.Int8).alias("low_confidence_flag"),
        ]
    )

    # disruption_flag: 1 if travel_time_var > µ + 2σ over trailing 168 h.
    df = df.with_columns(
        [
            pl.col("travel_time_var")
            .shift(1)
            .rolling_mean(window_size=168)
            .over("zone_id")
            .alias("_roll_mean_168"),
            pl.col("travel_time_var")
            .shift(1)
            .rolling_std(window_size=168)
            .over("zone_id")
            .alias("_roll_std_168"),
        ]
    ).with_columns(
        (
            pl.col("travel_time_var")
            > (pl.col("_roll_mean_168") + 2.0 * pl.col("_roll_std_168").fill_null(0.0))
        )
        .cast(pl.Int8)
        .alias("disruption_flag")
    ).drop(["_roll_mean_168", "_roll_std_168"])

    return df
