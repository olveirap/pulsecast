"""
tests/test_cache.py – Unit tests for the serving cache layer.

Covers:
  - pulsecast.serving.cache._bucket_congestion edge cases
  - pulsecast.serving.cache._make_key structure
  - pulsecast.serving.cache.ForecastCache get/set behaviour (Redis mocked)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pulsecast.serving.cache import ForecastCache, _bucket_congestion, _make_key

# ---------------------------------------------------------------------------
# _bucket_congestion
# ---------------------------------------------------------------------------


class TestBucketCongestion:
    def test_exact_multiple(self):
        assert _bucket_congestion(10.0) == pytest.approx(10.0)
        assert _bucket_congestion(20.0) == pytest.approx(20.0)
        assert _bucket_congestion(0.0) == pytest.approx(0.0)

    def test_rounds_to_nearest_ten(self):
        # Default bucket size is 10.0
        assert _bucket_congestion(14.0) == pytest.approx(10.0)
        assert _bucket_congestion(16.0) == pytest.approx(20.0)
        # 25.0 / 10.0 = 2.5. round(2.5) is 2 (round half to even).
        assert _bucket_congestion(25.0) == pytest.approx(20.0)
        # 35.0 / 10.0 = 3.5. round(3.5) is 4.
        assert _bucket_congestion(35.0) == pytest.approx(40.0)
        assert _bucket_congestion(24.9) == pytest.approx(20.0)

    def test_zero(self):
        assert _bucket_congestion(0.0) == pytest.approx(0.0)

    def test_negative_value(self):
        assert _bucket_congestion(-3.0) == pytest.approx(0.0)
        assert _bucket_congestion(-7.0) == pytest.approx(-10.0)

    def test_large_value(self):
        assert _bucket_congestion(999.0) == pytest.approx(1000.0)

    def test_custom_bucket_size(self):
        assert _bucket_congestion(1.0, bucket_size=1.0) == pytest.approx(1.0)
        assert _bucket_congestion(1.4, bucket_size=1.0) == pytest.approx(1.0)
        assert _bucket_congestion(1.6, bucket_size=1.0) == pytest.approx(2.0)

    def test_returns_float(self):
        result = _bucket_congestion(13.0)
        assert isinstance(result, float)

    def test_precision_capped_at_6_decimals(self):
        result = _bucket_congestion(10.1)
        assert result == round(result, 6)


# ---------------------------------------------------------------------------
# _make_key
# ---------------------------------------------------------------------------


def test_make_key_format():
    key = _make_key(132, 3, 13.0)
    # 13.0 bucketed to nearest 10.0 → 10.0
    assert key == "forecast:132:3:10.0"


def test_make_key_uses_bucketed_congestion():
    # Two values that bucket to the same 10.0-multiple must produce
    # identical cache keys. 1.0 and 2.0 both round to 0.0.
    assert _make_key(1, 1, 1.0) == _make_key(1, 1, 2.0)


def test_make_key_different_routes_differ():
    assert _make_key(1, 1, 10.0) != _make_key(2, 1, 10.0)


def test_make_key_different_horizons_differ():
    assert _make_key(1, 1, 10.0) != _make_key(1, 2, 10.0)


# ---------------------------------------------------------------------------
# ForecastCache – Redis mocked via sys.modules
# ---------------------------------------------------------------------------


def _make_cache(redis_client_mock: MagicMock) -> ForecastCache:
    """Return a ForecastCache whose internal Redis client is the given mock."""
    import importlib
    import sys

    redis_module_mock = MagicMock()
    redis_module_mock.from_url.return_value = redis_client_mock

    with patch.dict(sys.modules, {"redis": redis_module_mock}):
        import pulsecast.serving.cache as sc

        importlib.reload(sc)
        return sc.ForecastCache(redis_url="redis://localhost:6379/0", ttl=60)


class TestForecastCacheGet:
    def test_returns_none_on_cache_miss(self):
        redis_mock = MagicMock()
        redis_mock.get.return_value = None
        cache = _make_cache(redis_mock)
        assert cache.get(1, 1, 0.0) is None

    def test_returns_dict_on_cache_hit(self):
        payload = {"p10": [1.0], "p50": [2.0], "p90": [3.0]}
        redis_mock = MagicMock()
        redis_mock.get.return_value = json.dumps(payload)
        cache = _make_cache(redis_mock)
        result = cache.get(1, 1, 0.0)
        assert result == payload

    def test_uses_bucketed_key(self):
        redis_mock = MagicMock()
        redis_mock.get.return_value = None
        cache = _make_cache(redis_mock)
        cache.get(1, 2, 13.0)
        # 13.0 buckets to 10.0
        redis_mock.get.assert_called_once_with("forecast:1:2:10.0")


class TestForecastCacheSet:
    def test_calls_redis_set_with_ttl(self):
        redis_mock = MagicMock()
        cache = _make_cache(redis_mock)
        payload = {"p10": [10.0], "p50": [20.0], "p90": [30.0]}
        # 5.0 / 10.0 = 0.5. round(0.5) is 0.
        cache.set(5, 3, 5.0, payload)
        redis_mock.set.assert_called_once()
        args, kwargs = redis_mock.set.call_args
        assert args[0] == "forecast:5:3:0.0"
        assert json.loads(args[1]) == payload
        assert kwargs.get("ex") == 60

    def test_set_then_get_roundtrip(self):
        stored: dict[str, str] = {}

        def fake_set(key, value, ex=None):
            stored[key] = value

        def fake_get(key):
            return stored.get(key)

        redis_mock = MagicMock()
        redis_mock.set.side_effect = fake_set
        redis_mock.get.side_effect = fake_get

        cache = _make_cache(redis_mock)
        payload = {"p10": [5.0], "p50": [10.0], "p90": [15.0]}
        cache.set(1, 1, 10.0, payload)
        result = cache.get(1, 1, 10.0)
        assert result == payload
