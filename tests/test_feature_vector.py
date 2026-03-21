"""
test_feature_vector.py – Unit tests for pulsecast/serving/main.py feature-vector helpers.

Tests mock the TimescaleDB connection so no live database is required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Unit tests for feature-vector helpers.
# ---------------------------------------------------------------------------
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from pulsecast.serving.features import (
    N_FEATURES,
    build_feature_vector,
    build_static_features,
    fetch_congestion_history,
    fetch_demand_history,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_DEMAND = np.arange(168, dtype=np.float32)  # 168 h of demand (0..167)
_FULL_CONG = np.ones(168, dtype=np.float32) * 0.5  # flat 0.5 delay_index


def _make_psycopg2_mock(rows: list[tuple]) -> MagicMock:
    """Return a connection mock that yields *rows* from fetchall()."""
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


def _make_pool_mock(conn_mock: MagicMock) -> MagicMock:
    """Return a _db_pool mock whose getconn() yields *conn_mock*."""
    pool_mock = MagicMock()
    pool_mock.getconn.return_value = conn_mock
    return pool_mock


def _make_static(
    route_id: int = 1,
    origin_var: float = 0.0,
    dest_var: float = 0.0,
    origin_sample_count: int = 15,
    dest_sample_count: int = 15,
    demand: np.ndarray | None = None,
    duration: np.ndarray | None = None,
    origin_cong: np.ndarray | None = None,
    dest_cong: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper around build_static_features."""
    if demand is None:
        demand = _FULL_DEMAND
    if duration is None:
        duration = np.zeros_like(demand)
    if origin_cong is None:
        origin_cong = _FULL_CONG
    if dest_cong is None:
        dest_cong = _FULL_CONG
    return build_static_features(
        route_id, origin_var, dest_var, demand, duration, origin_cong, dest_cong, origin_sample_count, dest_sample_count
    )


# ---------------------------------------------------------------------------
# build_static_features – shape and content
# ---------------------------------------------------------------------------

class TestBuildStaticFeatures:
    def test_shape_with_full_history(self):
        static = _make_static(132, 0.5, 0.6)
        assert static.shape == (N_FEATURES,)

    def test_dtype_is_float32(self):
        static = _make_static(1, 0.0, 0.0)
        assert static.dtype == np.float32

    def test_basic_slots(self):
        static = _make_static(132, 2.5, 3.5)
        assert static[0] == pytest.approx(132.0)  # route_id
        assert static[1] == pytest.approx(0.0)    # horizon_hours placeholder
        assert static[2] == pytest.approx(2.5)    # delay_index (travel_time_var)
        assert static[38] == pytest.approx(3.5)   # dest delay_index

    def test_demand_lags_filled_with_full_history(self):
        static = _make_static(1, 0.0, 0.0)
        # lag_1h at index 16 → _FULL_DEMAND[-1] = 167
        assert static[16] == pytest.approx(167.0)
        # lag_168h at index 27 → _FULL_DEMAND[0] = 0
        assert static[27] == pytest.approx(0.0)

    def test_demand_lags_zero_when_empty(self):
        empty = np.empty(0, dtype=np.float32)
        static = _make_static(1, 0.0, 0.0, demand=empty)
        assert np.all(static[16:28] == 0.0)

    def test_rolling_mean_3h(self):
        static = _make_static(1, 0.0, 0.0)
        # rolling_mean_3h at index 28 = mean of last 3 values: 165, 166, 167
        assert static[28] == pytest.approx(np.mean([165.0, 166.0, 167.0]))

    def test_congestion_lag1_filled(self):
        cong = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        static = _make_static(1, 0.0, 0.0, origin_cong=cong, dest_cong=cong)
        assert static[39] == pytest.approx(0.3)  # delay_index_lag1
        assert static[44] == pytest.approx(0.3)  # dest delay_index_lag1

    def test_low_confidence_flag_set(self):
        static = _make_static(1, 0.0, 0.0, origin_sample_count=5, dest_sample_count=5)
        assert static[43] == 1.0
        assert static[48] == 1.0

    def test_low_confidence_flag_unset(self):
        static = _make_static(1, 0.0, 0.0, origin_sample_count=15, dest_sample_count=15)
        assert static[43] == 0.0
        assert static[48] == 0.0


# ---------------------------------------------------------------------------
# build_feature_vector – horizon + calendar fills
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:
    def test_shape_with_full_history(self):
        static = _make_static(132, 0.5, 0.6)
        vec = build_feature_vector(1, static)
        assert vec.shape == (1, N_FEATURES), f"Expected (1, {N_FEATURES}), got {vec.shape}"

    def test_shape_with_empty_history(self):
        empty = np.empty(0, dtype=np.float32)
        static = _make_static(1, 0.0, 0.0, demand=empty, origin_cong=empty, dest_cong=empty)
        vec = build_feature_vector(24, static)
        assert vec.shape == (1, N_FEATURES)

    def test_shape_with_partial_history(self):
        partial_demand = np.ones(10, dtype=np.float32) * 50.0
        partial_cong = np.ones(5, dtype=np.float32) * 1.2
        static = _make_static(42, 1.5, 1.6, demand=partial_demand, origin_cong=partial_cong, dest_cong=partial_cong)
        vec = build_feature_vector(12, static)
        assert vec.shape == (1, N_FEATURES)

    def test_dtype_is_float32(self):
        vec = build_feature_vector(1, _make_static(1, 0.0, 0.0))
        assert vec.dtype == np.float32

    def test_horizon_hours_filled(self):
        static = _make_static(132, 2.5, 3.5)
        vec = build_feature_vector(6, static)
        flat = vec.flatten()
        assert flat[0] == pytest.approx(132.0)  # route_id (from static)
        assert flat[1] == pytest.approx(6.0)    # horizon_hours
        assert flat[2] == pytest.approx(2.5)    # delay_index (from static)

    def test_demand_lags_preserved(self):
        static = _make_static(1, 0.0, 0.0)
        vec = build_feature_vector(1, static)
        flat = vec.flatten()
        assert flat[16] == pytest.approx(167.0)  # lag_1h
        assert flat[27] == pytest.approx(0.0)    # lag_168h

    def test_demand_lags_zero_when_empty(self):
        empty = np.empty(0, dtype=np.float32)
        static = _make_static(1, 0.0, 0.0, demand=empty, origin_cong=empty, dest_cong=empty)
        vec = build_feature_vector(1, static)
        flat = vec.flatten()
        assert np.all(flat[16:28] == 0.0)

    def test_rolling_mean_3h_preserved(self):
        static = _make_static(1, 0.0, 0.0)
        vec = build_feature_vector(1, static)
        flat = vec.flatten()
        assert flat[28] == pytest.approx(np.mean([165.0, 166.0, 167.0]))

    def test_congestion_lag1_preserved(self):
        cong = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        static = _make_static(1, 0.0, 0.0, origin_cong=cong, dest_cong=cong)
        vec = build_feature_vector(1, static)
        flat = vec.flatten()
        assert flat[39] == pytest.approx(0.3)  # delay_index_lag1

    def test_not_all_zeros_with_real_data(self):
        static = _make_static(5, 1.0, 1.0)
        vec = build_feature_vector(3, static)
        assert not np.all(vec == 0.0)

    def test_static_not_mutated(self):
        """build_feature_vector must not modify the passed static array."""
        static = _make_static(1, 0.0, 0.0)
        original = static.copy()
        build_feature_vector(12, static)
        np.testing.assert_array_equal(static, original)


# ---------------------------------------------------------------------------
# Month cyclical encoding – Dec and Jan should be adjacent
# ---------------------------------------------------------------------------

class TestMonthCyclicalEncoding:
    def _get_month_angles(self, month: int) -> tuple[float, float]:
        """Return (month_sin, month_cos) from scalar_calendar_features for a date in *month*."""
        from datetime import UTC, datetime

        from pulsecast.features.calendar import scalar_calendar_features

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
        """December (month=12) with (month-1) encoding should sit at angle 2π*(11/12). Compares sin/cos."""
        sin_dec, cos_dec = self._get_month_angles(12)
        expected_sin = math.sin(2 * math.pi * 11 / 12)
        expected_cos = math.cos(2 * math.pi * 11 / 12)
        assert sin_dec == pytest.approx(expected_sin, abs=1e-5)
        assert cos_dec == pytest.approx(expected_cos, abs=1e-5)

    def test_january_at_angle_zero(self):
        """January (month=1) with (month-1) encoding should sit at angle 0. Compares sin/cos."""
        sin_jan, cos_jan = self._get_month_angles(1)
        assert sin_jan == pytest.approx(0.0, abs=1e-5)
        assert cos_jan == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# fetch_demand_history – DB interaction
# ---------------------------------------------------------------------------

class TestFetchDemandHistory:
    def test_returns_array_oldest_first(self):
        # DB returns newest-first: rows are (volume, duration)
        rows = [(float(168 - i), 100.0) for i in range(1, 169)]
        conn_mock = _make_psycopg2_mock(rows)
        # Pass MagicMock() as db_pool
        vols, durs = fetch_demand_history(MagicMock(getconn=MagicMock(return_value=conn_mock)), 132, n_hours=168)
        assert vols.shape == (168,)
        assert durs.shape == (168,)
        assert vols.dtype == np.float32
        # After reversal, index 0 should be the oldest (smallest) value
        assert float(vols[0]) == pytest.approx(0.0)
        assert float(vols[-1]) == pytest.approx(167.0)

    def test_returns_empty_on_db_error(self):
        pool_mock = MagicMock()
        pool_mock.getconn.side_effect = Exception("DB down")
        vols, durs = fetch_demand_history(pool_mock, 1)
        assert len(vols) == 0
        assert len(durs) == 0

    def test_returns_empty_when_no_rows(self):
        conn_mock = _make_psycopg2_mock([])
        vols, durs = fetch_demand_history(MagicMock(getconn=MagicMock(return_value=conn_mock)), 999)
        assert len(vols) == 0
        assert len(durs) == 0


# ---------------------------------------------------------------------------
# fetch_congestion_history – DB interaction
# ---------------------------------------------------------------------------

class TestFetchCongestionHistory:
    def test_returns_array_oldest_first(self):
        rows = [(float(i) * 0.1,) for i in range(168, 0, -1)]  # newest-first
        conn_mock = _make_psycopg2_mock(rows)
        result = fetch_congestion_history(MagicMock(getconn=MagicMock(return_value=conn_mock)), 132, n_hours=168)
        assert result.shape == (168,)
        assert result.dtype == np.float32
        # Oldest value should be index 0 (smallest)
        assert float(result[0]) <= float(result[-1])

    def test_returns_empty_on_db_error(self):
        pool_mock = MagicMock()
        pool_mock.getconn.side_effect = Exception("DB down")
        result = fetch_congestion_history(pool_mock, 1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_returns_empty_when_no_rows(self):
        conn_mock = _make_psycopg2_mock([])
        result = fetch_congestion_history(MagicMock(getconn=MagicMock(return_value=conn_mock)), 999)
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
        demand_rows = [(float(i), 100.0) for i in range(168, 0, -1)]  # newest-first
        cong_rows = [(0.5,)] * 168

        demand_conn = _make_psycopg2_mock(demand_rows)
        cong_conn = _make_psycopg2_mock(cong_rows)

        call_count = 0

        def _getconn_side_effect():
            nonlocal call_count
            call_count += 1
            return demand_conn if call_count == 1 else cong_conn

        pool_mock = MagicMock()
        pool_mock.getconn.side_effect = _getconn_side_effect

        v_hist, d_hist = fetch_demand_history(pool_mock, 132)
        c_hist = fetch_congestion_history(pool_mock, 132)

        static = build_static_features(132, 1.2, 1.3, v_hist, d_hist, c_hist, c_hist, 15, 15)
        vec = build_feature_vector(24, static)
        assert vec.shape == (1, N_FEATURES)
        assert vec.dtype == np.float32
        # Feature vector must not be all zeros
        assert not np.all(vec == 0.0)

