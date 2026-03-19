"""Shared configuration for pipeline entrypoint scripts."""

from __future__ import annotations

LGBM_FEATURES = [
    "route_id",
    "delay_index",
    "hour_of_day",
    "dow",
    "month",
    "week_of_year",
    "is_weekend",
    "days_to_next_us_holiday",
    "nyc_event_flag",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "ewm_trend_24h",
    "yoy_ratio",
    "delay_index_lag1",
    "delay_index_rolling3h",
    "disruption_flag",
]
