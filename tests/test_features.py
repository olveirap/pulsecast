"""
tests/test_features.py – Unit tests for feature builders.

Covers:
  - features.demand.build_demand_features
  - features.calendar.build_calendar_features
  - features.congestion.build_congestion_features
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from features.calendar import build_calendar_features
from features.congestion import build_congestion_features
from features.demand import build_demand_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOUR_START = datetime(2024, 3, 17, 0, 0, 0)  # St. Patrick's Day (NYC event)


def _demand_df(n: int = 30, route_ids: list[int] | None = None) -> pl.DataFrame:
    """Build a minimal demand DataFrame with *n* hourly rows per route."""
    if route_ids is None:
        route_ids = [1]
    rows = []
    for rid in route_ids:
        for i in range(n):
            rows.append(
                {
                    "route_id": rid,
                    "hour": datetime(2024, 1, 1, i % 24, 0, 0),
                    "volume": float(i + 1),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_demand_features
# ---------------------------------------------------------------------------


def test_demand_output_columns():
    df = build_demand_features(_demand_df())
    expected = {
        "route_id",
        "hour",
        "volume",
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
    }
    assert expected.issubset(set(df.columns))


def test_demand_row_count_preserved():
    raw = _demand_df(n=50)
    result = build_demand_features(raw)
    assert result.height == raw.height


def test_demand_lag_1h_is_previous_volume():
    """lag_1h at row i should equal volume at row i-1 (within the same route)."""
    df = build_demand_features(_demand_df(n=10))
    # First row lag should be null, rest should equal shifted volume.
    lag_col = df["lag_1h"].to_list()
    vol_col = df["volume"].to_list()
    assert lag_col[0] is None
    assert lag_col[1] == vol_col[0]
    assert lag_col[2] == vol_col[1]


def test_demand_no_cross_route_leakage():
    """Lag features for route 2 must not bleed from route 1."""
    df = build_demand_features(_demand_df(n=5, route_ids=[1, 2]))
    for rid in [1, 2]:
        sub = df.filter(pl.col("route_id") == rid)
        assert sub["lag_1h"][0] is None, f"First lag_1h for route {rid} must be null"


def test_demand_yoy_ratio_clipped():
    """yoy_ratio must stay within [0.01, 100]."""
    df = build_demand_features(_demand_df(n=50))
    ratios = df["yoy_ratio"].drop_nulls()
    assert (ratios >= 0.01).all()
    assert (ratios <= 100.0).all()


def test_demand_no_private_columns_remain():
    """Internal helper columns (prefixed _) must be dropped."""
    df = build_demand_features(_demand_df())
    private = [c for c in df.columns if c.startswith("_")]
    assert private == []


# ---------------------------------------------------------------------------
# build_calendar_features
# ---------------------------------------------------------------------------


def _calendar_df(hours: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"hour": hours})


def test_calendar_output_columns():
    df = build_calendar_features(_calendar_df([_HOUR_START]))
    for col in ("dow", "hour_of_day", "week_of_year", "days_to_next_us_holiday", "nyc_event_flag"):
        assert col in df.columns, f"Missing column: {col}"


def test_calendar_dow_range():
    # Polars dt.weekday() returns 1 (Monday) … 7 (Sunday).
    hours = [datetime(2024, 1, d, 0) for d in range(1, 8)]
    df = build_calendar_features(_calendar_df(hours))
    assert df["dow"].min() >= 1
    assert df["dow"].max() <= 7


def test_calendar_hour_of_day_range():
    hours = [datetime(2024, 1, 1, h) for h in range(24)]
    df = build_calendar_features(_calendar_df(hours))
    assert df["hour_of_day"].min() == 0
    assert df["hour_of_day"].max() == 23


def test_calendar_nyc_event_flag_st_patricks():
    """March 17 should be flagged as an NYC event."""
    df = build_calendar_features(_calendar_df([datetime(2024, 3, 17, 10)]))
    assert df["nyc_event_flag"][0] == 1


def test_calendar_nyc_event_flag_regular_day():
    """A random midweek date should not be flagged."""
    df = build_calendar_features(_calendar_df([datetime(2024, 6, 5, 9)]))
    assert df["nyc_event_flag"][0] == 0


def test_calendar_days_to_holiday_non_negative():
    hours = [datetime(2024, 1, d, 0) for d in range(1, 15)]
    df = build_calendar_features(_calendar_df(hours))
    assert (df["days_to_next_us_holiday"] >= 0).all()


def test_calendar_row_count_preserved():
    hours = [datetime(2024, 1, 1, h) for h in range(24)]
    raw = _calendar_df(hours)
    result = build_calendar_features(raw)
    assert result.height == raw.height


# ---------------------------------------------------------------------------
# build_congestion_features
# ---------------------------------------------------------------------------


def _congestion_df(n: int = 20, zone_ids: list[int] | None = None) -> pl.DataFrame:
    if zone_ids is None:
        zone_ids = [1]
    rows = []
    for zid in zone_ids:
        for i in range(n):
            rows.append(
                {
                    "zone_id": zid,
                    "hour": datetime(2024, 1, 1, i % 24, 0, 0),
                    "delay_index": float(i % 5) * 0.4,
                }
            )
    return pl.DataFrame(rows)


def test_congestion_output_columns():
    df = build_congestion_features(_congestion_df())
    for col in ("delay_index_lag1", "delay_index_rolling3h", "disruption_flag"):
        assert col in df.columns, f"Missing column: {col}"


def test_congestion_lag1_is_previous_value():
    df = build_congestion_features(_congestion_df(n=5))
    lag = df["delay_index_lag1"].to_list()
    raw = df["delay_index"].to_list()
    assert lag[0] is None
    assert lag[1] == pytest.approx(raw[0])


def test_congestion_disruption_flag_binary():
    df = build_congestion_features(_congestion_df(n=30))
    flags = df["disruption_flag"].to_list()
    assert all(f in (0, 1, None) for f in flags)


def test_congestion_no_cross_zone_leakage():
    df = build_congestion_features(_congestion_df(n=5, zone_ids=[10, 20]))
    for zid in [10, 20]:
        sub = df.filter(pl.col("zone_id") == zid)
        assert sub["delay_index_lag1"][0] is None


def test_congestion_no_private_columns():
    df = build_congestion_features(_congestion_df())
    private = [c for c in df.columns if c.startswith("_")]
    assert private == []


def test_congestion_row_count_preserved():
    raw = _congestion_df(n=15)
    result = build_congestion_features(raw)
    assert result.height == raw.height
