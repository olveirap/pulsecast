"""Shared configuration for pipeline entrypoint scripts."""

from __future__ import annotations

LGBM_FEATURES = [
    "route_id",
    "origin_travel_time_var",
    "dest_travel_time_var",
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
    "origin_delay_index_lag1",
    "origin_delay_index_rolling3h",
    "origin_disruption_flag",
    "dest_delay_index_lag1",
    "dest_delay_index_rolling3h",
    "dest_disruption_flag",
]