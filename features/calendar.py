"""
calendar.py – Calendar and event features for a given datetime index.

Features produced:
  - dow            (0=Monday … 6=Sunday)
  - hour_of_day    (0–23)
  - week_of_year   (ISO, 1–53)
  - days_to_next_us_holiday  (integer days until the next US federal holiday)
  - nyc_event_flag (1 if the date is a known NYC major-event date, else 0)
"""

from __future__ import annotations

import math
from datetime import date, datetime

import polars as pl

try:
    from holidays import country_holidays  # holidays >= 0.25
except ImportError:
    country_holidays = None  # type: ignore[assignment]

# A small hard-coded sample of recurring NYC major events.
# In production this would be populated from a data source.
_NYC_EVENTS: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 1),   # New Year's Day (parade/celebrations)
        (3, 17),  # St. Patrick's Day Parade
        (7, 4),   # Fourth of July fireworks
        (11, 1),  # NYC Marathon (first Sunday in November – approximate)
        (12, 31), # New Year's Eve
    }
)


def _us_holidays_for_year(year: int) -> set[date]:
    if country_holidays is None:
        return set()
    return set(country_holidays("US", years=year).keys())


def _days_to_next_holiday(dt: date) -> int:
    """Return calendar days from *dt* to the next US federal holiday."""
    holidays_this_year = _us_holidays_for_year(dt.year)
    holidays_next_year = _us_holidays_for_year(dt.year + 1)
    all_holidays = sorted(holidays_this_year | holidays_next_year)
    for h in all_holidays:
        if h >= dt:
            return (h - dt).days
    return 0


def scalar_calendar_features(dt: datetime) -> dict[str, float]:
    """Return all calendar features for a single *datetime* as a plain dict.

    Keys returned (in order):
      hour_of_day, dow, month, week_of_year, is_weekend,
      days_to_next_us_holiday, nyc_event_flag,
      hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    """
    d = dt.date()
    h = dt.hour
    dow = dt.weekday()  # 0 = Monday, 6 = Sunday
    month = dt.month
    week = dt.isocalendar()[1]
    return {
        "hour_of_day": float(h),
        "dow": float(dow),
        "month": float(month),
        "week_of_year": float(week),
        "is_weekend": float(int(dow >= 5)),
        "days_to_next_us_holiday": float(_days_to_next_holiday(d)),
        "nyc_event_flag": float(int((d.month, d.day) in _NYC_EVENTS)),
        "hour_sin": math.sin(2 * math.pi * h / 24),
        "hour_cos": math.cos(2 * math.pi * h / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def build_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Must contain a column ``hour`` of type Datetime (any precision).

    Returns
    -------
    pl.DataFrame
        Original columns plus calendar feature columns.
    """
    df = df.with_columns(
        [
            pl.col("hour").dt.weekday().alias("dow"),
            pl.col("hour").dt.hour().alias("hour_of_day"),
            pl.col("hour").dt.week().alias("week_of_year"),
        ]
    )

    # days_to_next_us_holiday – computed in Python then joined back.
    dates_col: list[date] = (
        df.select(pl.col("hour").dt.date().alias("d"))["d"]
        .to_list()
    )
    days_to_holiday = [_days_to_next_holiday(d) for d in dates_col]
    nyc_event = [
        int((d.month, d.day) in _NYC_EVENTS) for d in dates_col
    ]

    df = df.with_columns(
        [
            pl.Series("days_to_next_us_holiday", days_to_holiday, dtype=pl.Int32),
            pl.Series("nyc_event_flag", nyc_event, dtype=pl.Int8),
        ]
    )

    return df
