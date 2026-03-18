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
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from pulsecast.serving.cache import ForecastCache
from pulsecast.serving.schemas import ForecastRequest, ForecastResponse

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


def _build_feature_vector(route_id: int, horizon_hours: int, delay_index: float) -> np.ndarray:
    """
    Construct a simple feature vector for ONNX inference.

    In production this would call the feature store; here we return a
    placeholder vector of the correct shape.
    """
    features = np.zeros(_N_FEATURES, dtype=np.float32)
    features[0] = float(route_id)
    features[1] = float(horizon_hours)
    features[2] = delay_index
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

    # 1. Fetch delay_index
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

    # 3. Run ONNX inference (one prediction per horizon step)
    p10_list, p50_list, p90_list = [], [], []
    for h in range(1, horizon_hours + 1):
        features = _build_feature_vector(request.route_id, h, delay_index)
        preds = _run_onnx(features)
        p10_list.append(preds["p10"])
        p50_list.append(preds["p50"])
        p90_list.append(preds["p90"])

    payload: dict[str, Any] = {
        "p10": p10_list,
        "p50": p50_list,
        "p90": p90_list,
    }

    # 4. Store in cache
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
