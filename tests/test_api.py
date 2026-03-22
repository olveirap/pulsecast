"""
tests/test_api.py – FastAPI endpoint tests with mocked ONNX models and DB.

All heavy external dependencies (Redis, TimescaleDB, onnxruntime) are fully
mocked so that the tests run offline without any infrastructure.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers to build mock modules
# ---------------------------------------------------------------------------


def _make_redis_module() -> tuple[MagicMock, MagicMock]:
    """Return (redis_module_mock, redis_client_mock)."""
    redis_client = MagicMock()
    redis_client.get.return_value = None  # default: cache miss
    redis_module = MagicMock()
    redis_module.from_url.return_value = redis_client
    return redis_module, redis_client


def _make_ort_module() -> MagicMock:
    """Return a mock onnxruntime module with canned InferenceSession."""

    def _fake_session(path: str) -> MagicMock:
        sess = MagicMock()
        sess.get_inputs.return_value = [MagicMock(name="X")]

        def _fake_run(outputs, inputs):
            x = list(inputs.values())[0]
            return [np.ones((len(x), 1), dtype=np.float32)]

        sess.run.side_effect = _fake_run
        return sess

    ort_module = MagicMock()
    ort_module.InferenceSession.side_effect = _fake_session
    return ort_module


def _make_pool_mock() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (pool_class_mock, pool_instance_mock, cursor_mock).

    The pool_class_mock is used to patch ``psycopg2.pool.ThreadedConnectionPool``.
    ``pool_instance_mock.getconn()`` returns a connection whose cursor yields
    ``cursor_mock``.
    """
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (0.5, 10)
    mock_cursor.fetchall.return_value = [
        (132, 10, 20),
        (5, 10, 20),
        (1, 10, 20),
        (7, 10, 20)
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.minconn = 2
    mock_pool.maxconn = 10
    # Mirror the private attrs that the /health endpoint reads for pool stats.
    mock_pool._pool = []
    mock_pool._used = {}
    mock_pool.getconn.return_value = mock_conn

    mock_pool_class = MagicMock(return_value=mock_pool)
    return mock_pool_class, mock_pool, mock_cursor


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client():
    """
    Yield (TestClient, pulsecast.serving.main module, redis_client) with all external I/O mocked:
      - onnxruntime   → MagicMock InferenceSession returning fixed predictions
      - redis         → MagicMock client (cache always misses by default)
      - psycopg2.pool.ThreadedConnectionPool → MagicMock pool returning delay_index=0.5
    """
    redis_module, redis_client = _make_redis_module()
    ort_module = _make_ort_module()
    mock_pool_class, _mock_pool, _mock_cursor = _make_pool_mock()

    # Remove any previously cached serving modules.
    for mod_name in list(sys.modules):
        if mod_name.startswith("serving"):
            del sys.modules[mod_name]

    # Force N_FEATURES to 54 for tests to match the expected length of _FEATURE_NAMES
    with patch.dict(os.environ, {"N_FEATURES": "54"}):
        with (
            patch.dict(sys.modules, {"redis": redis_module, "onnxruntime": ort_module}),
        ):
            import pulsecast.serving.main as main_mod

            importlib.reload(main_mod)

            with (
                patch.object(main_mod.pg_pool, "ThreadedConnectionPool", new=mock_pool_class),
                patch("pulsecast.serving.main.fetch_bus_congestion", return_value=(0.5, 15)),
                patch("pulsecast.serving.main.fetch_demand_history", return_value=(np.zeros(168, dtype=np.float32), np.zeros(168, dtype=np.float32))),
                patch("pulsecast.serving.main.fetch_congestion_history", return_value=np.zeros(168, dtype=np.float32)),
            ):
                with TestClient(main_mod.app, raise_server_exceptions=False) as client:
                    yield client, main_mod, redis_client

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_ok(app_client):
    client, _, _rc = app_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_includes_pool_stats(app_client):
    """Health response must include db_pool stats when the pool is active."""
    client, _, _rc = app_client
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "db_pool" in data
    pool = data["db_pool"]
    assert "min" in pool
    assert "max" in pool
    assert "available" in pool
    assert "in_use" in pool


# ---------------------------------------------------------------------------
# POST /forecast – happy path
# ---------------------------------------------------------------------------


def test_forecast_returns_200_for_valid_request(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": 132, "horizon": 1})
    assert resp.status_code == 200


def test_forecast_response_schema(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": 132, "horizon": 1})
    data = resp.json()
    assert "route_id" in data
    assert "horizon" in data
    assert "p10" in data
    assert "p50" in data
    assert "p90" in data


def test_forecast_echoes_route_id_and_horizon(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": 5, "horizon": 3})
    data = resp.json()
    assert data["route_id"] == 5
    assert data["horizon"] == 3


def test_forecast_p_list_length_equals_horizon_times_24(app_client):
    """Each quantile list must have horizon × 24 entries."""
    client, _, _rc = app_client
    horizon = 2
    resp = client.post("/forecast", json={"route_id": 1, "horizon": horizon})
    data = resp.json()
    assert len(data["p10"]) == horizon * 24
    assert len(data["p50"]) == horizon * 24
    assert len(data["p90"]) == horizon * 24


def test_forecast_includes_latency_header(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": 1, "horizon": 1})
    assert "X-Latency-Ms" in resp.headers


# ---------------------------------------------------------------------------
# POST /forecast – validation errors
# ---------------------------------------------------------------------------


def test_forecast_rejects_horizon_zero(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": 1, "horizon": 0})
    assert resp.status_code == 422


def test_forecast_rejects_horizon_too_large(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": 1, "horizon": 8})
    assert resp.status_code == 422


def test_forecast_rejects_negative_route_id(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={"route_id": -1, "horizon": 1})
    assert resp.status_code == 422


def test_forecast_rejects_missing_body(app_client):
    client, _, _rc = app_client
    resp = client.post("/forecast", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /forecast – cache hit path
# ---------------------------------------------------------------------------


def test_forecast_serves_cached_result(app_client):
    """When the cache returns a payload, it must be reflected in the response."""
    client, _main_mod, redis_client = app_client
    cached_payload: dict[str, Any] = {"p10": [9.9], "p50": [19.9], "p90": [29.9]}
    redis_client.get.return_value = json.dumps(cached_payload)

    resp = client.post("/forecast", json={"route_id": 7, "horizon": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["p10"] == [9.9]
    assert data["p50"] == [19.9]
    assert data["p90"] == [29.9]


# ---------------------------------------------------------------------------
# GET /calibration
# ---------------------------------------------------------------------------


def test_calibration_returns_404_when_file_missing(app_client, tmp_path):
    """GET /calibration must return 404 when calibration.json does not exist."""
    client, main_mod, _ = app_client
    missing = tmp_path / "no_such_file.json"

    with patch.object(main_mod, "_CALIBRATION_PATH", missing):
        resp = client.get("/calibration")

    assert resp.status_code == 404
    assert "run evaluation first" in resp.json()["detail"].lower()


def test_calibration_returns_200_with_valid_file(app_client, tmp_path):
    """GET /calibration must return 200 and the parsed JSON when the file exists."""
    client, main_mod, _ = app_client
    calib_file = tmp_path / "calibration.json"
    calib_file.write_text(
        json.dumps({"nominal": [0.1, 0.5, 0.9], "observed": [0.08, 0.47, 0.93]})
    )

    with patch.object(main_mod, "_CALIBRATION_PATH", calib_file):
        resp = client.get("/calibration")

    assert resp.status_code == 200
    data = resp.json()
    assert data["nominal"] == pytest.approx([0.1, 0.5, 0.9])
    assert data["observed"] == pytest.approx([0.08, 0.47, 0.93])


def test_calibration_returns_500_on_malformed_file(app_client, tmp_path):
    """GET /calibration must return 500 with a JSON-specific message on invalid JSON."""
    client, main_mod, _ = app_client
    bad_file = tmp_path / "calibration.json"
    bad_file.write_text("not valid json {{{")

    with patch.object(main_mod, "_CALIBRATION_PATH", bad_file):
        resp = client.get("/calibration")

    assert resp.status_code == 500
    assert "malformed json" in resp.json()["detail"].lower()


def test_calibration_returns_500_on_schema_mismatch(app_client, tmp_path):
    """GET /calibration must return 500 with a schema message when JSON is valid but wrong shape."""
    client, main_mod, _ = app_client
    bad_file = tmp_path / "calibration.json"
    bad_file.write_text(json.dumps({"wrong_key": [1, 2, 3]}))

    with patch.object(main_mod, "_CALIBRATION_PATH", bad_file):
        resp = client.get("/calibration")

    assert resp.status_code == 500
    assert "schema" in resp.json()["detail"].lower()
