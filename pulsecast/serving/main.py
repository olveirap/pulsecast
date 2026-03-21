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
from pulsecast.serving.features import (
    N_FEATURES,
    build_feature_matrix,
    fetch_bus_congestion,
    fetch_congestion_history,
    fetch_demand_history,
    fetch_subway_delay,
    get_conn,
)
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

# Env-var override: reject mismatches at startup rather than silently
_N_FEATURES_ENV = int(os.getenv("N_FEATURES", str(N_FEATURES)))
if _N_FEATURES_ENV != N_FEATURES:
    raise RuntimeError(
        f"N_FEATURES={_N_FEATURES_ENV} (from env) but N_FEATURES is {N_FEATURES}; "
        "update N_FEATURES to match."
    )

_ROUTES_MAP: dict[int, tuple[int, int]] = {}

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_db_pool: pg_pool.ThreadedConnectionPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _db_pool, _ROUTES_MAP
    _db_pool = pg_pool.ThreadedConnectionPool(
        minconn=_DB_POOL_MIN,
        maxconn=_DB_POOL_MAX,
        dsn=_DB_DSN,
    )
    logger.info("DB pool created (min=%d, max=%d)", _db_pool.minconn, _db_pool.maxconn)

    # Load routes mapping into memory
    try:
        with get_conn(_db_pool) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT route_id, origin_zone_id, destination_zone_id FROM routes")
                _ROUTES_MAP = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        logger.info("Loaded %d routes from database", len(_ROUTES_MAP))
    except Exception:
        logger.exception("Failed to load routes from database")

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
            status_code=500,
            detail="Calibration file schema does not match expected format",
        ) from exc


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, raw_request: Request) -> Response:
    t0 = time.perf_counter()
    horizon_hours = request.horizon * 24

    # Resolve route_id to origin/destination zones
    zones = _ROUTES_MAP.get(request.route_id)
    if not zones:
        raise HTTPException(
            status_code=404, detail=f"Route ID {request.route_id} not found"
        )
    origin_id, dest_id = zones

    # 1. Fetch live congestion and subway data
    origin_var, origin_sample = fetch_bus_congestion(_db_pool, origin_id)
    dest_var, dest_sample = fetch_bus_congestion(_db_pool, dest_id)

    # 2. Check cache (using origin travel_time_var as part of key)
    cached = _cache.get(request.route_id, request.horizon, origin_var)
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

    # 3. Fetch historical time series
    demand_history, duration_history = fetch_demand_history(_db_pool, request.route_id)
    origin_history = fetch_congestion_history(_db_pool, origin_id)
    dest_history = fetch_congestion_history( _db_pool, dest_id)

    # 4. Build batch feature matrix and run inference
    features = build_feature_matrix(
        request.route_id,
        horizon_hours,
        origin_var,
        dest_var,
        origin_sample,
        dest_sample,
        demand_history,
        duration_history,
        origin_history,
        dest_history,
    )
    preds = _run_onnx(features)

    payload: dict[str, Any] = {
        "p10": preds["p10"],
        "p50": preds["p50"],
        "p90": preds["p90"],
    }

    # 5. Store in cache
    _cache.set(request.route_id, request.horizon, origin_var, payload)

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
