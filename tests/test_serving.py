"""
tests/test_serving.py – Unit tests for the Pulsecast serving module.

All ONNX sessions and database calls are mocked so that the tests run
offline without any models on disk or a running TimescaleDB.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_session(n_rows: int) -> MagicMock:
    """Return a fake onnxruntime.InferenceSession for *n_rows* output rows."""
    sess = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "X"
    sess.get_inputs.return_value = [mock_input]
    # Simulate returning one prediction per input row.
    sess.run.side_effect = lambda _out, feed: [
        np.ones(n_rows, dtype=np.float32) * 1.0
    ]
    return sess

# ---------------------------------------------------------------------------
# _build_feature_matrix
# ---------------------------------------------------------------------------

def test_build_feature_matrix_shape():
    import pulsecast.serving.main as m

    mat = m._build_feature_matrix(
        route_id=10, horizon_hours=24, travel_time_var=0.5, sample_count=15
    )
    assert mat.shape == (24, m._N_FEATURES)
    assert mat.dtype == np.float32

def test_build_feature_matrix_route_id_column():
    import pulsecast.serving.main as m

    mat = m._build_feature_matrix(
        route_id=42, horizon_hours=5, travel_time_var=0.0, sample_count=15
    )
    assert np.all(mat[:, 0] == 42.0)

def test_build_feature_matrix_horizon_steps():
    """Column 1 must be the per-step horizon (1 … horizon_hours)."""
    import pulsecast.serving.main as m

    horizon_hours = 6
    mat = m._build_feature_matrix(
        route_id=1, horizon_hours=horizon_hours, travel_time_var=0.0, sample_count=15
    )
    expected = np.arange(1, horizon_hours + 1, dtype=np.float32)
    np.testing.assert_array_equal(mat[:, 1], expected)

def test_build_feature_matrix_delay_index_column():
    """Column 2 now stores travel_time_var (aliased as delay_index)."""
    import pulsecast.serving.main as m

    mat = m._build_feature_matrix(
        route_id=1, horizon_hours=4, travel_time_var=3.7, sample_count=15
    )
    np.testing.assert_allclose(mat[:, 2], 3.7)

def test_build_feature_matrix_7day_rows():
    """A 7-day horizon must yield exactly 168 rows."""
    import pulsecast.serving.main as m

    mat = m._build_feature_matrix(
        route_id=1, horizon_hours=168, travel_time_var=0.0, sample_count=15
    )
    assert mat.shape[0] == 168

def test_build_feature_matrix_low_confidence_flag():
    import pulsecast.serving.main as m

    # sample_count < 10 should set index 42 to 1.0
    mat = m._build_feature_matrix(
        route_id=1, horizon_hours=1, travel_time_var=0.0, sample_count=5
    )
    assert mat[0, 42] == 1.0

    # sample_count >= 10 should set index 42 to 0.0
    mat = m._build_feature_matrix(
        route_id=1, horizon_hours=1, travel_time_var=0.0, sample_count=10
    )
    assert mat[0, 42] == 0.0

# ---------------------------------------------------------------------------
# _run_onnx
# ---------------------------------------------------------------------------

def test_run_onnx_calls_each_session_once():
    """_run_onnx must call each quantile session exactly once."""
    import pulsecast.serving.main as m

    horizon_hours = 168
    features = np.zeros((horizon_hours, m._N_FEATURES), dtype=np.float32)

    mock_sessions = {
        "p10": _make_mock_session(horizon_hours),
        "p50": _make_mock_session(horizon_hours),
        "p90": _make_mock_session(horizon_hours),
    }

    with patch.object(m, "_sessions", mock_sessions):
        result = m._run_onnx(features)

    for sess in mock_sessions.values():
        sess.run.assert_called_once()

    assert set(result.keys()) == {"p10", "p50", "p90"}
    assert len(result["p10"]) == horizon_hours
    assert len(result["p50"]) == horizon_hours
    assert len(result["p90"]) == horizon_hours

def test_run_onnx_raises_when_no_sessions():
    """_run_onnx must raise HTTP 503 when _sessions is empty."""
    from fastapi import HTTPException

    import pulsecast.serving.main as m

    features = np.zeros((24, m._N_FEATURES), dtype=np.float32)
    with patch.object(m, "_sessions", {}):
        with pytest.raises(HTTPException) as exc_info:
            m._run_onnx(features)
    assert exc_info.value.status_code == 503

# ---------------------------------------------------------------------------
# /forecast endpoint – exactly 3 ONNX calls for a 7-day request
# ---------------------------------------------------------------------------

def test_forecast_7day_makes_exactly_3_onnx_calls():
    """A 7-day forecast must result in exactly 3 ONNX session.run calls."""
    import pulsecast.serving.main as m

    horizon_hours = 7 * 24  # 168
    mock_sessions = {
        "p10": _make_mock_session(horizon_hours),
        "p50": _make_mock_session(horizon_hours),
        "p90": _make_mock_session(horizon_hours),
    }

    empty = np.empty(0, dtype=np.float32)
    with (
        patch.object(m, "_sessions", mock_sessions),
        patch.object(m, "_fetch_bus_congestion", return_value=(0.0, 15)),
        patch.object(m, "_fetch_subway_delay", return_value=0.0),
        patch.object(m, "_fetch_demand_history", return_value=empty),
        patch.object(m, "_fetch_congestion_history", return_value=empty),
        patch.object(m._cache, "get", return_value=None),
        patch.object(m._cache, "set"),
    ):
        client = TestClient(m.app)
        resp = client.post("/forecast", json={"route_id": 1, "horizon": 7})

    assert resp.status_code == 200
    total_run_calls = sum(sess.run.call_count for sess in mock_sessions.values())
    assert total_run_calls == 3, f"Expected 3 ONNX calls, got {total_run_calls}"

def test_forecast_response_length_matches_horizon():
    """Response lists must have length == horizon * 24."""
    import pulsecast.serving.main as m

    for horizon in (1, 3, 7):
        horizon_hours = horizon * 24
        mock_sessions = {
            "p10": _make_mock_session(horizon_hours),
            "p50": _make_mock_session(horizon_hours),
            "p90": _make_mock_session(horizon_hours),
        }

        empty = np.empty(0, dtype=np.float32)
        with (
            patch.object(m, "_sessions", mock_sessions),
            patch.object(m, "_fetch_bus_congestion", return_value=(0.0, 15)),
            patch.object(m, "_fetch_subway_delay", return_value=0.0),
            patch.object(m, "_fetch_demand_history", return_value=empty),
            patch.object(m, "_fetch_congestion_history", return_value=empty),
            patch.object(m._cache, "get", return_value=None),
            patch.object(m._cache, "set"),
        ):
            client = TestClient(m.app)
            resp = client.post("/forecast", json={"route_id": 1, "horizon": horizon})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["p10"]) == horizon_hours
        assert len(body["p50"]) == horizon_hours
        assert len(body["p90"]) == horizon_hours
