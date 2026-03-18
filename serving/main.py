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

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from features.calendar import scalar_calendar_features
from serving.cache import ForecastCache
from serving.schemas import ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ONNX_DIR = Path(os.getenv("ONNX_DIR", "models/onnx"))
_DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://pulsecast:pulsecast@timescaledb:5432/pulsecast",
)
_N_FEATURES = int(os.getenv("N_FEATURES", "42"))

# Canonical ordered feature names – must stay in sync with _build_feature_vector.
_FEATURE_NAMES: list[str] = [
    # ── Basic (3) ──────────────────────────────────────────────────────────
    "route_id",
    "horizon_hours",
    "delay_index",
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
    # ── Congestion (4) ─────────────────────────────────────────────────────
    "delay_index_lag1",
    "delay_index_lag24",
    "delay_index_rolling3h",
    "disruption_flag",
]

assert len(_FEATURE_NAMES) == _N_FEATURES, (
    f"Expected {_N_FEATURES} feature names, got {len(_FEATURE_NAMES)}"
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Pulsecast", version="0.1.0")

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

def _fetch_delay_index(route_id: int) -> float:
    """
    Retrieve the latest delay_index for *route_id* from Redis (via cache)
    or fall back to TimescaleDB.
    """
    try:
        conn = psycopg2.connect(_DB_DSN)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT delay_index
                    FROM delay_index
                    WHERE zone_id = %s
                    ORDER BY hour DESC
                    LIMIT 1
                    """,
                    (route_id,),
                )
                row = cur.fetchone()
        conn.close()
        if row:
            return float(row[0])
    except Exception:
        logger.exception("Failed to fetch delay_index from TimescaleDB")
    return 0.0


def _fetch_demand_history(route_id: int, n_hours: int = 168) -> np.ndarray:
    """
    Fetch the last *n_hours* of hourly demand volumes for *route_id* from
    the ``demand`` TimescaleDB hypertable.

    Returns a float32 array sorted oldest-first (index 0 = oldest hour,
    index -1 = most recent hour).  Returns an empty array on error.
    """
    try:
        conn = psycopg2.connect(_DB_DSN)
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
        conn.close()
        if rows:
            # rows are newest-first; reverse to oldest-first
            return np.array([float(r[0]) for r in reversed(rows)], dtype=np.float32)
    except Exception:
        logger.exception("Failed to fetch demand history from TimescaleDB")
    return np.empty(0, dtype=np.float32)


def _fetch_congestion_history(route_id: int, n_hours: int = 168) -> np.ndarray:
    """
    Fetch the last *n_hours* of delay_index values for zone *route_id* from
    the ``delay_index`` TimescaleDB hypertable.

    Returns a float32 array sorted oldest-first (index 0 = oldest hour,
    index -1 = most recent hour).  Returns an empty array on error.
    """
    try:
        conn = psycopg2.connect(_DB_DSN)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT delay_index
                    FROM delay_index
                    WHERE zone_id = %s
                    ORDER BY hour DESC
                    LIMIT %s
                    """,
                    (route_id, n_hours),
                )
                rows = cur.fetchall()
        conn.close()
        if rows:
            # rows are newest-first; reverse to oldest-first
            return np.array([float(r[0]) for r in reversed(rows)], dtype=np.float32)
    except Exception:
        logger.exception("Failed to fetch congestion history from TimescaleDB")
    return np.empty(0, dtype=np.float32)


def _build_feature_vector(
    route_id: int,
    horizon_hours: int,
    delay_index: float,
    demand_history: np.ndarray,
    congestion_history: np.ndarray,
) -> np.ndarray:
    """
    Construct a 42-dimensional feature vector for ONNX inference.

    Feature layout matches *_FEATURE_NAMES* (see module constant):
      [0-2]   Basic: route_id, horizon_hours, delay_index
      [3-15]  Calendar (target hour): hour_of_day, dow, month, week_of_year,
              is_weekend, days_to_next_us_holiday, nyc_event_flag,
              hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
      [16-27] Demand lags (1,2,3,6,12,24,48,72,96,120,144,168 h)
      [28-34] Rolling means of demand (3,6,12,24,48,72,168 h)
      [35-36] EWM trend of demand (span=24, span=168)
      [37]    YoY ratio (0.0 when < 8760 h of history is available)
      [38-41] Congestion: delay_index_lag1, delay_index_lag24,
              delay_index_rolling3h, disruption_flag

    Parameters
    ----------
    route_id : int
    horizon_hours : int
        Number of hours ahead being predicted.
    delay_index : float
        Current (live) delay_index for the zone.
    demand_history : np.ndarray
        Float32 array of demand volumes sorted oldest-first (up to 168 values).
        Missing values default to 0.0.
    congestion_history : np.ndarray
        Float32 array of delay_index values sorted oldest-first (up to 168 values).
        Missing values default to 0.0.

    Returns
    -------
    np.ndarray of shape (1, N_FEATURES)
    """
    features = np.zeros(_N_FEATURES, dtype=np.float32)

    # ── Basic ──────────────────────────────────────────────────────────────
    features[0] = float(route_id)
    features[1] = float(horizon_hours)
    features[2] = float(delay_index)

    # ── Calendar features for the target prediction hour ───────────────────
    target_dt = datetime.now(tz=UTC) + timedelta(hours=horizon_hours)
    cal = scalar_calendar_features(target_dt)
    cal_keys = [
        "hour_of_day", "dow", "month", "week_of_year", "is_weekend",
        "days_to_next_us_holiday", "nyc_event_flag",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    for i, key in enumerate(cal_keys):
        features[3 + i] = float(cal[key])

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
        # else: already 0.0 from np.zeros initialisation

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
    # else: already 0.0

    # ── YoY ratio – 0.0 when fewer than 8760 h of history are available ───
    # features[37] stays 0.0 (sentinel value; model trained to handle it)

    # ── Congestion features ────────────────────────────────────────────────
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
    # else: disruption_flag = 0.0

    # ── Validate output length ─────────────────────────────────────────────
    if features.shape[0] != _N_FEATURES:
        raise ValueError(
            f"Feature vector length {features.shape[0]} != N_FEATURES {_N_FEATURES}"
        )

    return features.reshape(1, -1)


def _run_onnx(features: np.ndarray) -> dict[str, float]:
    if not _sessions:
        raise HTTPException(status_code=503, detail="ONNX models not loaded.")
    results: dict[str, float] = {}
    for q_name, sess in _sessions.items():
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: features})[0].flatten()
        results[q_name] = float(out[0])
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, raw_request: Request) -> Response:
    t0 = time.perf_counter()
    horizon_hours = request.horizon * 24

    # 1. Fetch live delay_index
    delay_index = _fetch_delay_index(request.route_id)

    # 2. Check cache
    cached = _cache.get(request.route_id, request.horizon, delay_index)
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

    # 3. Fetch historical time series once (shared across all horizon steps)
    demand_history = _fetch_demand_history(request.route_id)
    congestion_history = _fetch_congestion_history(request.route_id)

    # 4. Run ONNX inference (one prediction per horizon step)
    p10_list, p50_list, p90_list = [], [], []
    for h in range(1, horizon_hours + 1):
        features = _build_feature_vector(
            request.route_id, h, delay_index, demand_history, congestion_history
        )
        preds = _run_onnx(features)
        p10_list.append(preds["p10"])
        p50_list.append(preds["p50"])
        p90_list.append(preds["p90"])

    payload: dict[str, Any] = {
        "p10": p10_list,
        "p50": p50_list,
        "p90": p90_list,
    }

    # 5. Store in cache
    _cache.set(request.route_id, request.horizon, delay_index, payload)

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
