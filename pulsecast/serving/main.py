"""
main.py – FastAPI application for Pulsecast probabilistic demand forecasting.

Endpoints
---------
POST /forecast
    Accept route_id + horizon, run ONNX inference, return p10/p50/p90.
GET  /health
    Liveness probe.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from psycopg2 import pool as pg_pool
from pydantic import ValidationError

from pulsecast.features.calendar import scalar_calendar_features
from pulsecast.serving.cache import ForecastCache
from pulsecast.serving.schemas import CalibrationResponse, ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ONNX_DIR = Path(os.getenv("ONNX_DIR", "models/onnx"))
_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@timescaledb:5432/pulsecast",
)
_CALIBRATION_PATH = Path(os.getenv("CALIBRATION_PATH", "data/results/calibration.json"))
_DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
_DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))

# Validate both are positive and DB_POOL_MAX >= DB_POOL_MIN
if _DB_POOL_MIN <= 0:
    raise ValueError(f"DB_POOL_MIN must be positive, got {_DB_POOL_MIN}") 
if _DB_POOL_MAX < _DB_POOL_MIN:
    raise ValueError(f"DB_POOL_MAX must be >= DB_POOL_MIN, got {_DB_POOL_MAX} < {_DB_POOL_MIN}")

# Canonical ordered feature names – must stay in sync with _build_feature_matrix.
_FEATURE_NAMES: list[str] = [
    # ── Basic (3) ──────────────────────────────────────────────────────────
    "route_id",
    "horizon_hours",
    "delay_index",  # Now refers to travel_time_var
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
    # ── Congestion (5) ─────────────────────────────────────────────────────
    "delay_index_lag1",      # travel_time_var lag1
    "delay_index_lag24",     # travel_time_var lag24
    "delay_index_rolling3h", # travel_time_var rolling3h
    "disruption_flag",
    "low_confidence_flag",   # sample_count < 10
]

_N_FEATURES: int = len(_FEATURE_NAMES)

# Env-var override: reject mismatches at startup rather than silently
_N_FEATURES_ENV = int(os.getenv("N_FEATURES", str(_N_FEATURES)))
if _N_FEATURES_ENV != _N_FEATURES:
    raise RuntimeError(
        f"N_FEATURES={_N_FEATURES_ENV} (from env) but _FEATURE_NAMES has "
        f"{_N_FEATURES} entries; update N_FEATURES or _FEATURE_NAMES to match."
    )

# Slice bounds for the calendar feature block within _FEATURE_NAMES.
_CALENDAR_START: int = 3   # first calendar feature index
_CALENDAR_END: int = 16    # one past the last calendar feature index (13 features)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_db_pool: pg_pool.ThreadedConnectionPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _db_pool
    _db_pool = pg_pool.ThreadedConnectionPool(
        minconn=_DB_POOL_MIN,
        maxconn=_DB_POOL_MAX,
        dsn=_DB_DSN,
    )
    logger.info("DB pool created (min=%d, max=%d)", _db_pool.minconn, _db_pool.maxconn)
    try:
        yield
    finally:
        if _db_pool is not None:
            _db_pool.closeall()
        _db_pool = None
        logger.info("DB pool closed")


app = FastAPI(title="Pulsecast", version="0.1.0", lifespan=lifespan)

_cache = ForecastCache()

try:
    import onnxruntime as ort

    _sessions: dict[str, ort.InferenceSession] = {
        "p10": ort.InferenceSession(str(_ONNX_DIR / "lgbm_p10.onnx")),
        "p50": ort.InferenceSession(str(_ONNX_DIR / "lgbm_p50.onnx")),
        "p90": ort.InferenceSession(str(_ONNX_DIR / "lgbm_p90.onnx")),
    }
    logger.info("ONNX models loaded from %s", _ONNX_DIR)
except Exception:
    logger.warning("ONNX models not found – inference will fail until exported.")
    _sessions = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _get_conn():
    """Yield a psycopg2 connection borrowed from the pool, returning it on exit."""
    if _db_pool is None:
        raise RuntimeError("DB pool not initialized")
    conn = _db_pool.getconn()
    try:
        yield conn
    finally:
        _db_pool.putconn(conn)


def _fetch_bus_congestion(zone_id: int) -> tuple[float, int]:
    """
    Retrieve the latest travel_time_var and sample_count for *zone_id* 
    from the congestion hypertable.
    """
    if _db_pool is None:
        return 0.0, 0
    try:
        with _get_conn() as conn:
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

def _fetch_subway_delay(zone_id: int) -> float:
    """
    Retrieve the latest mean_delay for *zone_id* aggregated across 
    relevant feeds from the subway_delay hypertable.
    """
    if _db_pool is None:
        return 0.0
    try:
        with _get_conn() as conn:
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

def _fetch_demand_history(route_id: int, n_hours: int = 168) -> np.ndarray:
    """
    Fetch the last *n_hours* of hourly demand volumes for *route_id* from
    the ``demand`` TimescaleDB hypertable.

    Returns a float32 array sorted oldest-first (index 0 = oldest hour,
    index -1 = most recent hour).  Returns an empty array on error.
    """
    if _db_pool is None:
        return np.empty(0, dtype=np.float32)
    try:
        with _get_conn() as conn:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT volume
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
            return np.array([float(r[0]) for r in reversed(rows)], dtype=np.float32)
    except Exception:
        logger.exception("Failed to fetch demand history from TimescaleDB")
    return np.empty(0, dtype=np.float32)

def _fetch_congestion_history(zone_id: int, n_hours: int = 168) -> np.ndarray:
    """
    Fetch the last *n_hours* of travel_time_var values for *zone_id* from
    the ``congestion`` TimescaleDB hypertable.

    Returns a float32 array sorted oldest-first (index 0 = oldest hour,
    index -1 = most recent hour).  Returns an empty array on error.
    """
    if _db_pool is None:
        return np.empty(0, dtype=np.float32)
    try:
        with _get_conn() as conn:
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

def _build_static_features(
    route_id: int,
    travel_time_var: float,
    demand_history: np.ndarray,
    congestion_history: np.ndarray,
    sample_count: int = 0,
) -> np.ndarray:
    """
    Build the feature slots that do not depend on *horizon_hours* or calendar.

    Populates indices:
      [0]     route_id
      [2]     delay_index (live travel_time_var)
      [16-27] Demand lags (1,2,3,6,12,24,48,72,96,120,144,168 h)
      [28-34] Rolling means of demand (3,6,12,24,48,72,168 h)
      [35-36] EWM trend of demand (span=24, span=168)
      [37]    YoY ratio (0.0 sentinel when < 8760 h of history)
      [38-42] Congestion: delay_index_lag1, delay_index_lag24,
              delay_index_rolling3h, disruption_flag, low_confidence_flag

    Indices 1 (horizon_hours) and 3-15 (calendar) are left at 0.0 for
    ``_build_feature_vector`` to fill per horizon step.

    Returns
    -------
    np.ndarray of shape (_N_FEATURES,)
    """
    features = np.zeros(_N_FEATURES, dtype=np.float32)

    # ── Basic (static part) ────────────────────────────────────────────────
    features[0] = float(route_id)
    features[2] = float(travel_time_var)

    # ── Demand lag features ────────────────────────────────────────────────
    n_d = len(demand_history)
    for i, lag in enumerate([1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]):
        idx = n_d - lag
        features[16 + i] = float(demand_history[idx]) if idx >= 0 else 0.0

    # ── Rolling means of demand ────────────────────────────────────────────
    for i, window in enumerate([3, 6, 12, 24, 48, 72, 168]):
        if n_d >= window:
            features[28 + i] = float(np.mean(demand_history[-window:]))
        elif n_d > 0:
            features[28 + i] = float(np.mean(demand_history))

    # ── EWM trend (span=24 and span=168, adjust=False) ────────────────────
    if n_d > 0:
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

    # ── YoY ratio – 0.0 sentinel when fewer than 8760 h of history ────────
    # features[37] stays 0.0

    # ── Congestion features ───────────────────────────────────────────────
    n_c = len(congestion_history)
    features[38] = float(congestion_history[-1]) if n_c >= 1 else 0.0  # lag1
    features[39] = float(congestion_history[-24]) if n_c >= 24 else 0.0  # lag24
    if n_c >= 3:
        features[40] = float(np.mean(congestion_history[-3:]))
    elif n_c > 0:
        features[40] = float(np.mean(congestion_history))
    if n_c >= 168:
        mean168 = float(np.mean(congestion_history[-168:]))
        std168 = float(np.std(congestion_history[-168:]))
        features[41] = 1.0 if float(congestion_history[-1]) > mean168 + 2.0 * std168 else 0.0
    
    features[42] = 1.0 if sample_count < 10 else 0.0

    return features

def _build_feature_vector(horizon_hours: int, static_features: np.ndarray) -> np.ndarray:
    """
    Assemble the final _N_FEATURES-dimensional feature vector for a single
    *horizon_hours* step by filling the horizon-dependent slots into a copy
    of *static_features*.

    Populates indices:
      [1]    horizon_hours
      [3-15] Calendar features for the target prediction hour (derived from
             _FEATURE_NAMES[_CALENDAR_START:_CALENDAR_END])

    Returns
    -------
    np.ndarray of shape (1, _N_FEATURES)
    """
    features = static_features.copy()
    features[1] = float(horizon_hours)

    # Calendar features – keys derived from _FEATURE_NAMES to stay in sync.
    target_dt = datetime.now(tz=UTC) + timedelta(hours=horizon_hours)
    cal = scalar_calendar_features(target_dt)
    for i, key in enumerate(_FEATURE_NAMES[_CALENDAR_START:_CALENDAR_END]):
        features[_CALENDAR_START + i] = float(cal[key])

    if features.shape[0] != _N_FEATURES:
        raise ValueError(
            f"Feature vector length {features.shape[0]} != N_FEATURES {_N_FEATURES}"
        )

    return features.reshape(1, -1)

def _build_feature_matrix(
    route_id: int,
    horizon_hours: int,
    travel_time_var: float,
    sample_count: int,
    demand_history: np.ndarray | None = None,
    congestion_history: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct a batch feature matrix for all horizon steps in one shot.
    """
    if demand_history is None:
        demand_history = np.empty(0, dtype=np.float32)
    if congestion_history is None:
        congestion_history = np.empty(0, dtype=np.float32)

    static = _build_static_features(
        route_id,
        travel_time_var,
        demand_history,
        congestion_history,
        sample_count=sample_count,
    )
    rows = [_build_feature_vector(h, static).flatten() for h in range(1, horizon_hours + 1)]
    return np.array(rows, dtype=np.float32)

def _run_onnx(features: np.ndarray) -> dict[str, list[float]]:
    """Run each quantile ONNX session exactly once with the full batch matrix."""
    if not _sessions:
        raise HTTPException(status_code=503, detail="ONNX models not loaded.")
    results: dict[str, list[float]] = {}
    for q_name, sess in _sessions.items():
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: features})[0].flatten()
        results[q_name] = out.tolist()
    return results

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok"}
    if _db_pool is not None:
        try:
            result["db_pool"] = {
                "min": _db_pool.minconn,
                "max": _db_pool.maxconn,
                "available": len(_db_pool._pool),
                "in_use": len(_db_pool._used),
            }
        except Exception:
            result["db_pool"] = {"error": "stats unavailable"}
    return result


@app.get("/calibration", response_model=CalibrationResponse)
async def calibration() -> CalibrationResponse:
    """Return empirical quantile coverage from the held-out evaluation set."""
    if not _CALIBRATION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Calibration data unavailable – run evaluation first",
        )
    try:
        raw = _CALIBRATION_PATH.read_text()
    except OSError as exc:
        logger.error("Unable to read calibration file %s: %s", _CALIBRATION_PATH, exc)
        raise HTTPException(status_code=500, detail="Unable to read calibration file") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Malformed JSON in calibration file %s: %s", _CALIBRATION_PATH, exc)
        raise HTTPException(status_code=500, detail="Malformed JSON in calibration file") from exc
    try:
        return CalibrationResponse(**payload)
    except (ValidationError, TypeError) as exc:
        logger.error("Schema mismatch in calibration file %s: %s", _CALIBRATION_PATH, exc)
        raise HTTPException(
            status_code=500, detail="Calibration file schema does not match expected format"
        ) from exc

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, raw_request: Request) -> Response:
    t0 = time.perf_counter()
    horizon_hours = request.horizon * 24

    # 1. Fetch live congestion and subway data
    travel_time_var, sample_count = _fetch_bus_congestion(request.route_id)
    subway_delay = _fetch_subway_delay(request.route_id)
    
    # Log subway_delay for now as it's a supplementary signal
    if subway_delay > 0:
        logger.info("Zone %d has subway delay: %.2fs", request.route_id, subway_delay)

    # 2. Check cache (using travel_time_var as the key part)
    cached = _cache.get(request.route_id, request.horizon, travel_time_var)
    if cached is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        resp_content: dict[str, Any] = {
            "route_id": request.route_id,
            "horizon": request.horizon,
            **cached,
        }
        return JSONResponse(
            content=resp_content,
            headers={"X-Latency-Ms": f"{latency_ms:.1f}"},
        )

    # 3. Fetch historical time series once
    demand_history = _fetch_demand_history(request.route_id)
    congestion_history = _fetch_congestion_history(request.route_id)

    # 4. Build batch feature matrix and run inference
    features = _build_feature_matrix(
        request.route_id, 
        horizon_hours, 
        travel_time_var, 
        sample_count,
        demand_history, 
        congestion_history
    )
    preds = _run_onnx(features)

    payload: dict[str, Any] = {
        "p10": preds["p10"],
        "p50": preds["p50"],
        "p90": preds["p90"],
    }

    # 5. Store in cache
    _cache.set(request.route_id, request.horizon, travel_time_var, payload)

    latency_ms = (time.perf_counter() - t0) * 1000
    resp_content = {
        "route_id": request.route_id,
        "horizon": request.horizon,
        **payload,
    }
    return JSONResponse(
        content=resp_content,
        headers={"X-Latency-Ms": f"{latency_ms:.1f}"},
    )
