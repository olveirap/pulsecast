"""
test_feature_vector.py – Unit tests for serving/main.py feature-vector helpers.

Tests mock the TimescaleDB connection so no live database is required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers to avoid importing the whole FastAPI app (which tries to load ONNX)
# ---------------------------------------------------------------------------
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub out onnxruntime so the module-level try/except in main.py doesn't fail.
ort_stub = types.ModuleType("onnxruntime")
ort_stub.InferenceSession = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("onnxruntime", ort_stub)

# Stub out redis so ForecastCache.__init__ doesn't fail at import time.
redis_stub = types.ModuleType("redis")
redis_stub.from_url = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
sys.modules.setdefault("redis", redis_stub)

from serving.main import (  # noqa: E402
    _N_FEATURES,
    _build_feature_vector,
    _fetch_congestion_history,
    _fetch_demand_history,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_DEMAND = np.arange(168, dtype=np.float32)  # 168 h of demand (0..167)
_FULL_CONG = np.ones(168, dtype=np.float32) * 0.5  # flat 0.5 delay_index


def _make_psycopg2_mock(rows: list[tuple]) -> MagicMock:
    """Return a psycopg2.connect mock that yields *rows* from fetchall()."""
    cursor_mock = MagicMock()
    cursor_mock.__enter__ = MagicMock(return_value=cursor_mock)
    cursor_mock.__exit__ = MagicMock(return_value=False)
    cursor_mock.fetchall.return_value = rows
    cursor_mock.fetchone.return_value = rows[0] if rows else None

    conn_mock = MagicMock()
    conn_mock.__enter__ = MagicMock(return_value=conn_mock)
    conn_mock.__exit__ = MagicMock(return_value=False)
    conn_mock.cursor.return_value = cursor_mock

    return conn_mock


# ---------------------------------------------------------------------------
# _build_feature_vector – shape and type
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:
    def test_shape_with_full_history(self):
        vec = _build_feature_vector(132, 1, 0.5, _FULL_DEMAND, _FULL_CONG)
        assert vec.shape == (1, _N_FEATURES), f"Expected (1, {_N_FEATURES}), got {vec.shape}"

    def test_shape_with_empty_history(self):
        empty = np.empty(0, dtype=np.float32)
        vec = _build_feature_vector(1, 24, 0.0, empty, empty)
        assert vec.shape == (1, _N_FEATURES)

    def test_shape_with_partial_history(self):
        partial_demand = np.ones(10, dtype=np.float32) * 50.0
        partial_cong = np.ones(5, dtype=np.float32) * 1.2
        vec = _build_feature_vector(42, 12, 1.5, partial_demand, partial_cong)
        assert vec.shape == (1, _N_FEATURES)

    def test_dtype_is_float32(self):
        vec = _build_feature_vector(1, 1, 0.0, _FULL_DEMAND, _FULL_CONG)
        assert vec.dtype == np.float32

    def test_basic_features_filled(self):
        vec = _build_feature_vector(132, 6, 2.5, _FULL_DEMAND, _FULL_CONG)
        flat = vec.flatten()
        assert flat[0] == pytest.approx(132.0)   # route_id
        assert flat[1] == pytest.approx(6.0)     # horizon_hours
        assert flat[2] == pytest.approx(2.5)     # delay_index

    def test_demand_lags_filled_with_full_history(self):
        # With 168 rows, lag_1h should be demand_history[-1], lag_168h should be demand_history[0]
        vec = _build_feature_vector(1, 1, 0.0, _FULL_DEMAND, _FULL_CONG)
        flat = vec.flatten()
        # lag_1h is at index 16 → value should be _FULL_DEMAND[-1] = 167
        assert flat[16] == pytest.approx(167.0)
        # lag_168h is at index 27 → value should be _FULL_DEMAND[0] = 0
        assert flat[27] == pytest.approx(0.0)

    def test_demand_lags_zero_when_empty(self):
        empty = np.empty(0, dtype=np.float32)
        vec = _build_feature_vector(1, 1, 0.0, empty, empty)
        flat = vec.flatten()
        # All demand lag features (indices 16–27) should be 0
        assert np.all(flat[16:28] == 0.0)

    def test_rolling_means_filled_with_full_history(self):
        vec = _build_feature_vector(1, 1, 0.0, _FULL_DEMAND, _FULL_CONG)
        flat = vec.flatten()
        # rolling_mean_3h at index 28 = mean of last 3 values: 165, 166, 167
        assert flat[28] == pytest.approx(np.mean([165.0, 166.0, 167.0]))

    def test_congestion_lag1_filled(self):
        cong = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec = _build_feature_vector(1, 1, 0.0, _FULL_DEMAND, cong)
        flat = vec.flatten()
        assert flat[38] == pytest.approx(0.3)  # delay_index_lag1

    def test_not_all_zeros_with_real_data(self):
        vec = _build_feature_vector(5, 3, 1.0, _FULL_DEMAND, _FULL_CONG)
        # At minimum the basic features and calendar features should be non-zero
        assert not np.all(vec == 0.0)


# ---------------------------------------------------------------------------
# _fetch_demand_history – DB interaction
# ---------------------------------------------------------------------------

class TestFetchDemandHistory:
    def test_returns_array_oldest_first(self):
        # DB returns newest-first: rows are (167,), (166,), ..., (0,)
        rows = [(float(168 - i),) for i in range(1, 169)]  # 167, 166, ..., 0
        conn_mock = _make_psycopg2_mock(rows)
        with patch("serving.main.psycopg2.connect", return_value=conn_mock):
            result = _fetch_demand_history(132, n_hours=168)
        assert result.shape == (168,)
        assert result.dtype == np.float32
        # After reversal, index 0 should be the oldest (smallest) value
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[-1]) == pytest.approx(167.0)

    def test_returns_empty_on_db_error(self):
        with patch("serving.main.psycopg2.connect", side_effect=Exception("DB down")):
            result = _fetch_demand_history(1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_returns_empty_when_no_rows(self):
        conn_mock = _make_psycopg2_mock([])
        with patch("serving.main.psycopg2.connect", return_value=conn_mock):
            result = _fetch_demand_history(999)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _fetch_congestion_history – DB interaction
# ---------------------------------------------------------------------------

class TestFetchCongestionHistory:
    def test_returns_array_oldest_first(self):
        rows = [(float(i) * 0.1,) for i in range(168, 0, -1)]  # newest-first
        conn_mock = _make_psycopg2_mock(rows)
        with patch("serving.main.psycopg2.connect", return_value=conn_mock):
            result = _fetch_congestion_history(132, n_hours=168)
        assert result.shape == (168,)
        assert result.dtype == np.float32
        # Oldest value should be index 0 (smallest)
        assert float(result[0]) <= float(result[-1])

    def test_returns_empty_on_db_error(self):
        with patch("serving.main.psycopg2.connect", side_effect=Exception("DB down")):
            result = _fetch_congestion_history(1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_returns_empty_when_no_rows(self):
        conn_mock = _make_psycopg2_mock([])
        with patch("serving.main.psycopg2.connect", return_value=conn_mock):
            result = _fetch_congestion_history(999)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration: full feature vector via mocked DB
# ---------------------------------------------------------------------------

class TestFeatureVectorWithMockedDB:
    def test_shape_end_to_end(self):
        """
        Simulate the call sequence used in the /forecast endpoint:
        fetch history from mocked DB, then build feature vector.
        """
        demand_rows = [(float(i),) for i in range(168, 0, -1)]  # newest-first
        cong_rows = [(0.5,)] * 168

        demand_conn = _make_psycopg2_mock(demand_rows)
        cong_conn = _make_psycopg2_mock(cong_rows)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return demand_conn if call_count == 1 else cong_conn

        with patch("serving.main.psycopg2.connect", side_effect=_connect_side_effect):
            d_hist = _fetch_demand_history(132)
            c_hist = _fetch_congestion_history(132)

        vec = _build_feature_vector(132, 24, 1.2, d_hist, c_hist)
        assert vec.shape == (1, _N_FEATURES)
        assert vec.dtype == np.float32
        # Feature vector must not be all zeros
        assert not np.all(vec == 0.0)
