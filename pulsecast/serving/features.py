"""
features.py – Feature construction and data-fetching helpers for the serving layer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from pulsecast.features.calendar import scalar_calendar_features

if TYPE_CHECKING:
    from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

# Canonical ordered feature names – must stay in sync with build_feature_matrix.
_FEATURE_NAMES: list[str] = [
    # ── Basic (3) ──────────────────────────────────────────────────────────
    "route_id",
    "horizon_hours",
    "origin_delay_index",  # origin travel_time_var
    # ── Calendar – target prediction hour (13) ─────────────────────────────
    "hour_of_day",
    "dow",
    "month",
    "week_of_year",
    "is_weekend",
    "days_to_next_us_holiday",
    "nyc_event_flag",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # ── Demand lags (12) ───────────────────────────────────────────────────
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_6h",
    "lag_12h",
    "lag_24h",
    "lag_48h",
    "lag_72h",
    "lag_96h",
    "lag_120h",
    "lag_144h",
    "lag_168h",
    # ── Rolling means (7) ──────────────────────────────────────────────────
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_12h",
    "rolling_mean_24h",
    "rolling_mean_48h",
    "rolling_mean_72h",
    "rolling_mean_168h",
    # ── EWM trend (2) ──────────────────────────────────────────────────────
    "ewm_trend_24h",
    "ewm_trend_168h",
    # ── Year-over-year ratio (1) ────────────────────────────────────────────
    "yoy_ratio",
    # ── Congestion (11) ─────────────────────────────────────────────────────
    "dest_delay_index",
    "origin_delay_index_lag1",
    "origin_delay_index_lag24",
    "origin_delay_index_rolling3h",
    "origin_disruption_flag",
    "origin_low_confidence_flag",
    "dest_delay_index_lag1",
    "dest_delay_index_lag24",
    "dest_delay_index_rolling3h",
    "dest_disruption_flag",
    "dest_low_confidence_flag",
    "duration_lag_1h",
    "duration_lag_24h",
    "duration_lag_168h",
    "duration_rolling_mean_3h",
    "duration_rolling_mean_24h",
]

N_FEATURES: int = len(_FEATURE_NAMES)

# Slice bounds for the calendar feature block within _FEATURE_NAMES.
_CALENDAR_START: int = 3   # first calendar feature index
_CALENDAR_END: int = 16    # one past the last calendar feature index (13 features)


@contextmanager
def get_conn(db_pool: ThreadedConnectionPool | None):
    """Yield a psycopg2 connection borrowed from the pool, returning it on exit."""
    if db_pool is None:
        raise RuntimeError("DB pool not initialized")
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


def fetch_bus_congestion(db_pool: ThreadedConnectionPool | None, zone_id: int) -> tuple[float, int]:
    """
    Retrieve the latest travel_time_var and sample_count for *zone_id* 
    from the congestion hypertable.
    """
    if db_pool is None:
        return 0.0, 0
    try:
        with get_conn(db_pool) as conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT travel_time_var, sample_count
                        FROM congestion
                        WHERE zone_id = %s
                        ORDER BY hour DESC
                        LIMIT 1
                        """,
                        (zone_id,),
                    )
                    row = cur.fetchone()
        if row:
            return float(row[0]), int(row[1])
    except Exception:
        logger.exception("Failed to fetch congestion data from TimescaleDB")
    return 0.0, 0


def fetch_subway_delay(db_pool: ThreadedConnectionPool | None, zone_id: int) -> float:
    """
    Retrieve the latest mean_delay for *zone_id* aggregated across 
    relevant feeds from the subway_delay hypertable.
    """
    if db_pool is None:
        return 0.0
    try:
        with get_conn(db_pool) as conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT SUM(mean_delay * trip_count) / NULLIF(SUM(trip_count), 0)
                        FROM subway_delay
                        WHERE zone_id = %s
                          AND hour > NOW() - INTERVAL '1 hour'
                        """,
                        (zone_id,),
                    )
                    row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        logger.exception("Failed to fetch subway_delay from TimescaleDB")
    return 0.0


def fetch_demand_history(
    db_pool: ThreadedConnectionPool | None, route_id: int, n_hours: int = 168
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch the last *n_hours* of hourly demand volumes and average durations
    for *route_id* from the ``demand`` TimescaleDB hypertable.

    Returns (volumes, durations) as float32 arrays sorted oldest-first.
    Returns empty arrays on error.
    """
    if db_pool is None:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    try:
        with get_conn(db_pool) as conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT volume, avg_duration
                        FROM demand
                        WHERE route_id = %s
                        ORDER BY hour DESC
                        LIMIT %s
                        """,
                        (route_id, n_hours),
                    )
                    rows = cur.fetchall()
        if rows:
            # rows are newest-first; reverse to oldest-first
            vols = np.array([float(r[0]) for r in reversed(rows)], dtype=np.float32)
            durs = np.array([float(r[1] or 0.0) for r in reversed(rows)], dtype=np.float32)
            return vols, durs
    except Exception:
        logger.exception("Failed to fetch demand history from TimescaleDB")
    return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)


def fetch_congestion_history(db_pool: ThreadedConnectionPool | None, zone_id: int, n_hours: int = 168) -> np.ndarray:
    """
    Fetch the last *n_hours* of travel_time_var values for *zone_id* from
    the ``congestion`` TimescaleDB hypertable.

    Returns a float32 array sorted oldest-first (index 0 = oldest hour,
    index -1 = most recent hour).  Returns an empty array on error.
    """
    if db_pool is None:
        return np.empty(0, dtype=np.float32)
    try:
        with get_conn(db_pool) as conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT travel_time_var
                        FROM congestion
                        WHERE zone_id = %s
                        ORDER BY hour DESC
                        LIMIT %s
                        """,
                        (zone_id, n_hours),
                    )
                    rows = cur.fetchall()
        if rows:
            # rows are newest-first; reverse to oldest-first
            return np.array([float(r[0]) for r in reversed(rows)], dtype=np.float32)
    except Exception:
        logger.exception("Failed to fetch congestion history from TimescaleDB")
    return np.empty(0, dtype=np.float32)


def compute_congestion_history_features(
    history: np.ndarray,
    sample_count: int,
    feature_offset: int = 0,
) -> tuple[float, float, float, float, float]:
    """
    Compute five congestion-related features from a history array.
    """
    n = len(history)
    
    # Feature 0: lag1 - most recent travel_time_var
    lag1 = float(history[-1]) if n >= 1 else 0.0
    
    # Feature 1: lag24 - travel_time_var from 24 hours ago
    lag24 = float(history[-24]) if n >= 24 else 0.0
    
    # Feature 2: rolling_mean_3h - mean of last 3 hours (or all available)
    if n >= 3:
        rolling_mean_3h = float(np.mean(history[-3:]))
    elif n > 0:
        rolling_mean_3h = float(np.mean(history))
    else:
        rolling_mean_3h = 0.0
    
    # Feature 3: disruption_flag - 1 if lag1 exceeds mean + 2*std of last 168h
    if n >= 168:
        m = float(np.mean(history[-168:]))
        s = float(np.std(history[-168:]))
        disruption_flag = 1.0 if lag1 > m + 2.0 * s else 0.0
    else:
        disruption_flag = 0.0
    
    # Feature 4: low_confidence_flag - 1 if sample_count < 10
    low_confidence_flag = 1.0 if sample_count < 10 else 0.0
    
    return (lag1, lag24, rolling_mean_3h, disruption_flag, low_confidence_flag)


def build_static_features(
    route_id: int,
    origin_var: float,
    dest_var: float,
    demand_history: np.ndarray,
    duration_history: np.ndarray,
    origin_history: np.ndarray,
    dest_history: np.ndarray,
    origin_sample_count: int = 0,
    dest_sample_count: int = 0,
) -> np.ndarray:
    """
    Build the feature slots that do not depend on *horizon_hours* or calendar.
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)

    # ── Basic (static part) ────────────────────────────────────────────────
    features[0] = float(route_id)
    features[2] = float(origin_var)

    # ── Demand lag features (indices 16-27) ────────────────────────────────
    n_vol = len(demand_history)
    for i, lag in enumerate([1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]):
        idx = n_vol - lag
        features[16 + i] = float(demand_history[idx]) if idx >= 0 else 0.0

    # ── Rolling means of demand (indices 28-34) ───────────────────────────
    for i, window in enumerate([3, 6, 12, 24, 48, 72, 168]):
        if n_vol >= window:
            features[28 + i] = float(np.mean(demand_history[-window:]))
        elif n_vol > 0:
            features[28 + i] = float(np.mean(demand_history))

    # ── EWM trend (indices 35-36) ──────────────────────────────────────────
    if n_vol > 0:
        alpha24 = 2.0 / (24.0 + 1.0)
        alpha168 = 2.0 / (168.0 + 1.0)
        ewm24 = float(demand_history[0])
        ewm168 = float(demand_history[0])
        for v in demand_history[1:]:
            fv = float(v)
            ewm24 = alpha24 * fv + (1.0 - alpha24) * ewm24
            ewm168 = alpha168 * fv + (1.0 - alpha168) * ewm168
        features[35] = float(ewm24)
        features[36] = float(ewm168)

    # ── YoY ratio (index 37) stays 0.0 for now

    # ── Congestion features ───────────────────────────────────────────────
    features[38] = float(dest_var)

    # Origin history features (indices 39-43)
    origin_features = compute_congestion_history_features(origin_history, origin_sample_count, feature_offset=0)
    features[39:44] = origin_features

    # Destination history features (indices 44-48)
    dest_features = compute_congestion_history_features(dest_history, dest_sample_count, feature_offset=5)
    features[44:49] = dest_features

    # ── Duration features (indices 49-53) ──────────────────────────────────
    n_dur = len(duration_history)
    if n_dur > 0:
        # duration_lag_1h
        features[49] = float(duration_history[-1]) if n_dur >= 1 else 0.0
        # duration_lag_24h
        features[50] = float(duration_history[-24]) if n_dur >= 24 else 0.0
        # duration_lag_168h
        features[51] = float(duration_history[-168]) if n_dur >= 168 else 0.0
        # duration_rolling_mean_3h
        if n_dur >= 3:
            features[52] = float(np.mean(duration_history[-3:]))
        else:
            features[52] = float(np.mean(duration_history))
        # duration_rolling_mean_24h
        if n_dur >= 24:
            features[53] = float(np.mean(duration_history[-24:]))
        else:
            features[53] = float(np.mean(duration_history))

    return features


def build_feature_vector(horizon_hours: int, static_features: np.ndarray) -> np.ndarray:
    """
    Assemble the final N_FEATURES-dimensional feature vector for a single
    *horizon_hours* step.
    """
    features = static_features.copy()
    features[1] = float(horizon_hours)

    # Calendar features – keys derived from _FEATURE_NAMES to stay in sync.
    target_dt = datetime.now(tz=UTC) + timedelta(hours=horizon_hours)
    cal = scalar_calendar_features(target_dt)
    for i, key in enumerate(_FEATURE_NAMES[_CALENDAR_START:_CALENDAR_END]):
        features[_CALENDAR_START + i] = float(cal[key])

    return features.reshape(1, -1)


def build_feature_matrix(
    route_id: int,
    horizon_hours: int,
    origin_var: float,
    dest_var: float,
    origin_sample_count: int,
    dest_sample_count: int,
    demand_history: np.ndarray | None = None,
    duration_history: np.ndarray | None = None,
    origin_history: np.ndarray | None = None,
    dest_history: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct a batch feature matrix for all horizon steps in one shot.
    """
    if demand_history is None:
        demand_history = np.empty(0, dtype=np.float32)
    if duration_history is None:
        duration_history = np.empty(0, dtype=np.float32)
    if origin_history is None:
        origin_history = np.empty(0, dtype=np.float32)
    if dest_history is None:
        dest_history = np.empty(0, dtype=np.float32)

    static = build_static_features(
        route_id,
        origin_var,
        dest_var,
        demand_history,
        duration_history,
        origin_history,
        dest_history,
        origin_sample_count=origin_sample_count,
        dest_sample_count=dest_sample_count,
    )
    rows = [
        build_feature_vector(h, static).flatten() for h in range(1, horizon_hours + 1)
    ]
    return np.array(rows, dtype=np.float32)
