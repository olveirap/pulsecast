"""
congestion.py – Features derived from the delay_index congestion signal.

Features produced per (zone_id, hour):
  - delay_index_lag1      (delay_index at t-1 h)
  - delay_index_rolling3h (mean over previous 3 hours)
  - disruption_flag       (1 if delay_index > µ + 2σ over the trailing 168 h window)
"""

from __future__ import annotations

import polars as pl


def build_congestion_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: ``zone_id`` (int), ``hour`` (Datetime),
        ``delay_index`` (float).  Rows must be sorted by (zone_id, hour).

    Returns
    -------
    pl.DataFrame
        Original columns plus congestion feature columns.
    """
    df = df.sort(["zone_id", "hour"])

    df = df.with_columns(
        [
            # lag-1 delay_index
            pl.col("delay_index")
            .shift(1)
            .over("zone_id")
            .alias("delay_index_lag1"),
            # rolling 3-hour mean (excluding current observation)
            pl.col("delay_index")
            .shift(1)
            .rolling_mean(window_size=3)
            .over("zone_id")
            .alias("delay_index_rolling3h"),
        ]
    )

    # disruption_flag: 1 if delay_index > µ + 2σ over trailing 168 h.
    df = df.with_columns(
        [
            pl.col("delay_index")
            .shift(1)
            .rolling_mean(window_size=168)
            .over("zone_id")
            .alias("_roll_mean_168"),
            pl.col("delay_index")
            .shift(1)
            .rolling_std(window_size=168)
            .over("zone_id")
            .alias("_roll_std_168"),
        ]
    ).with_columns(
        (
            pl.col("delay_index")
            > (pl.col("_roll_mean_168") + 2.0 * pl.col("_roll_std_168").fill_null(0.0))
        )
        .cast(pl.Int8)
        .alias("disruption_flag")
    ).drop(["_roll_mean_168", "_roll_std_168"])

    return df
