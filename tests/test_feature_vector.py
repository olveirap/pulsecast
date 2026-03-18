"""
test_feature_vector.py – Unit tests for serving/main.py feature-vector helpers.

Tests mock the TimescaleDB connection so no live database is required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers to avoid importing the whole FastAPI app (which tries to load ONNX)
# ---------------------------------------------------------------------------
import math
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
    _build_static_features,
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


def _make_static(
    route_id: int = 1,
    delay_index: float = 0.0,
    demand: np.ndarray | None = None,
    cong: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper around _build_static_features."""
    if demand is None:
        demand = _FULL_DEMAND
    if cong is None:
        cong = _FULL_CONG
    return _build_static_features(route_id, delay_index, demand, cong)


# ---------------------------------------------------------------------------
# _build_static_features – shape and content
# ---------------------------------------------------------------------------

class TestBuildStaticFeatures:
    def test_shape_with_full_history(self):
        static = _build_static_features(132, 0.5, _FULL_DEMAND, _FULL_CONG)
        assert static.shape == (_N_FEATURES,)

    def test_dtype_is_float32(self):
        static = _build_static_features(1, 0.0, _FULL_DEMAND, _FULL_CONG)
        assert static.dtype == np.float32

    def test_basic_slots(self):
        static = _build_static_features(132, 2.5, _FULL_DEMAND, _FULL_CONG)
        assert static[0] == pytest.approx(132.0)  # route_id
        assert static[1] == pytest.approx(0.0)    # horizon_hours placeholder
        assert static[2] == pytest.approx(2.5)    # delay_index

    def test_demand_lags_filled_with_full_history(self):
        static = _build_static_features(1, 0.0, _FULL_DEMAND, _FULL_CONG)
        # lag_1h at index 16 → _FULL_DEMAND[-1] = 167
        assert static[16] == pytest.approx(167.0)
        # lag_168h at index 27 → _FULL_DEMAND[0] = 0
        assert static[27] == pytest.approx(0.0)

    def test_demand_lags_zero_when_empty(self):
        empty = np.empty(0, dtype=np.float32)
        static = _build_static_features(1, 0.0, empty, empty)
        assert np.all(static[16:28] == 0.0)

    def test_rolling_mean_3h(self):
        static = _build_static_features(1, 0.0, _FULL_DEMAND, _FULL_CONG)
        # rolling_mean_3h at index 28 = mean of last 3 values: 165, 166, 167
        assert static[28] == pytest.approx(np.mean([165.0, 166.0, 167.0]))

    def test_congestion_lag1_filled(self):
        cong = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        static = _build_static_features(1, 0.0, _FULL_DEMAND, cong)
        assert static[38] == pytest.approx(0.3)  # delay_index_lag1


# ---------------------------------------------------------------------------
# _build_feature_vector – horizon + calendar fills
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:
    def test_shape_with_full_history(self):
        static = _make_static(132, 0.5)
        vec = _build_feature_vector(1, static)
        assert vec.shape == (1, _N_FEATURES), f"Expected (1, {_N_FEATURES}), got {vec.shape}"

    def test_shape_with_empty_history(self):
        empty = np.empty(0, dtype=np.float32)
        static = _make_static(1, 0.0, demand=empty, cong=empty)
        vec = _build_feature_vector(24, static)
        assert vec.shape == (1, _N_FEATURES)

    def test_shape_with_partial_history(self):
        partial_demand = np.ones(10, dtype=np.float32) * 50.0
        partial_cong = np.ones(5, dtype=np.float32) * 1.2
        static = _make_static(42, 1.5, demand=partial_demand, cong=partial_cong)
        vec = _build_feature_vector(12, static)
        assert vec.shape == (1, _N_FEATURES)

    def test_dtype_is_float32(self):
        vec = _build_feature_vector(1, _make_static(1, 0.0))
        assert vec.dtype == np.float32

    def test_horizon_hours_filled(self):
        static = _make_static(132, 2.5)
        vec = _build_feature_vector(6, static)
        flat = vec.flatten()
        assert flat[0] == pytest.approx(132.0)  # route_id (from static)
        assert flat[1] == pytest.approx(6.0)    # horizon_hours
        assert flat[2] == pytest.approx(2.5)    # delay_index (from static)

    def test_demand_lags_preserved(self):
        static = _make_static(1, 0.0)
        vec = _build_feature_vector(1, static)
        flat = vec.flatten()
        assert flat[16] == pytest.approx(167.0)  # lag_1h
        assert flat[27] == pytest.approx(0.0)    # lag_168h

    def test_demand_lags_zero_when_empty(self):
        empty = np.empty(0, dtype=np.float32)
        static = _make_static(1, 0.0, demand=empty, cong=empty)
        vec = _build_feature_vector(1, static)
        flat = vec.flatten()
        assert np.all(flat[16:28] == 0.0)

    def test_rolling_mean_3h_preserved(self):
        static = _make_static(1, 0.0)
        vec = _build_feature_vector(1, static)
        flat = vec.flatten()
        assert flat[28] == pytest.approx(np.mean([165.0, 166.0, 167.0]))

    def test_congestion_lag1_preserved(self):
        cong = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        static = _make_static(1, 0.0, cong=cong)
        vec = _build_feature_vector(1, static)
        flat = vec.flatten()
        assert flat[38] == pytest.approx(0.3)  # delay_index_lag1

    def test_not_all_zeros_with_real_data(self):
        static = _make_static(5, 1.0)
        vec = _build_feature_vector(3, static)
        assert not np.all(vec == 0.0)

    def test_static_not_mutated(self):
        """_build_feature_vector must not modify the passed static array."""
        static = _make_static(1, 0.0)
        original = static.copy()
        _build_feature_vector(12, static)
        np.testing.assert_array_equal(static, original)


# ---------------------------------------------------------------------------
# Month cyclical encoding – Dec and Jan should be adjacent
# ---------------------------------------------------------------------------

class TestMonthCyclicalEncoding:
    def _get_month_angles(self, month: int) -> tuple[float, float]:
        """Return (month_sin, month_cos) from scalar_calendar_features for a date in *month*."""
        from datetime import UTC, datetime

        from features.calendar import scalar_calendar_features

        dt = datetime(2025, month, 15, 12, 0, 0, tzinfo=UTC)
        cal = scalar_calendar_features(dt)
        return cal["month_sin"], cal["month_cos"]

    def test_january_is_adjacent_to_december(self):
        """With (month - 1) encoding, Dec angle ≈ Jan - one step."""
        sin_jan, cos_jan = self._get_month_angles(1)
        sin_dec, cos_dec = self._get_month_angles(12)
        # Euclidean distance between Dec and Jan should be one step
        dist = math.sqrt((sin_jan - sin_dec) ** 2 + (cos_jan - cos_dec) ** 2)
        # chord length for one step: 2 * sin(π/12)
        expected_chord = 2 * math.sin(math.pi / 12)
        assert dist == pytest.approx(expected_chord, abs=1e-5)

    def test_december_wraps_to_zero_angle(self):
        """December (month=12) with (month-1) encoding should sit at angle 2π*(11/12)."""
        sin_dec, cos_dec = self._get_month_angles(12)
        expected_sin = math.sin(2 * math.pi * 11 / 12)
        expected_cos = math.cos(2 * math.pi * 11 / 12)
        assert sin_dec == pytest.approx(expected_sin, abs=1e-5)
        assert cos_dec == pytest.approx(expected_cos, abs=1e-5)

    def test_january_at_angle_zero(self):
        """January (month=1) with (month-1) encoding should sit at angle 0."""
        sin_jan, cos_jan = self._get_month_angles(1)
        assert sin_jan == pytest.approx(0.0, abs=1e-5)
        assert cos_jan == pytest.approx(1.0, abs=1e-5)


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
        fetch history from mocked DB, pre-compute static features,
        then build the per-step feature vector.
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

        static = _build_static_features(132, 1.2, d_hist, c_hist)
        vec = _build_feature_vector(24, static)
        assert vec.shape == (1, _N_FEATURES)
        assert vec.dtype == np.float32
        # Feature vector must not be all zeros
        assert not np.all(vec == 0.0)
