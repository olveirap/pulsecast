"""
tests/test_cache.py – Unit tests for the serving cache layer.

Covers:
  - pulsecast.serving.cache._bucket_delay edge cases
  - pulsecast.serving.cache._make_key structure
  - pulsecast.serving.cache.ForecastCache get/set behaviour (Redis mocked)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pulsecast.serving.cache import ForecastCache, _bucket_delay, _make_key

# ---------------------------------------------------------------------------
# _bucket_delay
# ---------------------------------------------------------------------------


class TestBucketDelay:
    def test_exact_multiple(self):
        assert _bucket_delay(1.0) == pytest.approx(1.0)
        assert _bucket_delay(1.5) == pytest.approx(1.5)
        assert _bucket_delay(2.0) == pytest.approx(2.0)

    def test_rounds_to_nearest_half(self):
        assert _bucket_delay(1.24) == pytest.approx(1.0)
        assert _bucket_delay(1.26) == pytest.approx(1.5)
        assert _bucket_delay(1.75) == pytest.approx(2.0)
        assert _bucket_delay(1.74) == pytest.approx(1.5)

    def test_zero(self):
        assert _bucket_delay(0.0) == pytest.approx(0.0)

    def test_negative_value(self):
        # Negative delay indices should still bucket correctly.
        assert _bucket_delay(-0.3) == pytest.approx(-0.5)
        assert _bucket_delay(-0.1) == pytest.approx(0.0)

    def test_large_value(self):
        assert _bucket_delay(99.9) == pytest.approx(100.0)

    def test_custom_bucket_size(self):
        assert _bucket_delay(1.0, bucket_size=1.0) == pytest.approx(1.0)
        assert _bucket_delay(1.4, bucket_size=1.0) == pytest.approx(1.0)
        assert _bucket_delay(1.6, bucket_size=1.0) == pytest.approx(2.0)

    def test_returns_float(self):
        result = _bucket_delay(1.3)
        assert isinstance(result, float)

    def test_precision_capped_at_6_decimals(self):
        # _bucket_delay rounds to 6 decimal places; result must equal itself rounded.
        result = _bucket_delay(0.1)
        assert result == round(result, 6)


# ---------------------------------------------------------------------------
# _make_key
# ---------------------------------------------------------------------------


def test_make_key_format():
    key = _make_key(132, 3, 1.3)
    # 1.3 bucketed to nearest 0.5 → 1.5
    assert key == "forecast:132:3:1.5"


def test_make_key_uses_bucketed_delay():
    # Two delay indices that bucket to the same 0.5-multiple must produce
    # identical cache keys.  0.1 and 0.2 both round to 0.0.
    assert _make_key(1, 1, 0.1) == _make_key(1, 1, 0.2)


def test_make_key_different_routes_differ():
    assert _make_key(1, 1, 1.0) != _make_key(2, 1, 1.0)


def test_make_key_different_horizons_differ():
    assert _make_key(1, 1, 1.0) != _make_key(1, 2, 1.0)


# ---------------------------------------------------------------------------
# ForecastCache – Redis mocked via sys.modules
# ---------------------------------------------------------------------------


def _make_cache(redis_client_mock: MagicMock) -> ForecastCache:
    """Return a ForecastCache whose internal Redis client is the given mock.

    ``redis`` is imported lazily inside ``__init__``, so we patch it via
    ``sys.modules`` rather than ``patch("pulsecast.serving.cache.redis")``.
    """
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
        cache.get(1, 2, 1.3)
        # delay 1.3 buckets to 1.5
        redis_mock.get.assert_called_once_with("forecast:1:2:1.5")


class TestForecastCacheSet:
    def test_calls_redis_set_with_ttl(self):
        redis_mock = MagicMock()
        cache = _make_cache(redis_mock)
        payload = {"p10": [10.0], "p50": [20.0], "p90": [30.0]}
        cache.set(5, 3, 0.5, payload)
        redis_mock.set.assert_called_once()
        args, kwargs = redis_mock.set.call_args
        assert args[0] == "forecast:5:3:0.5"
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
        cache.set(1, 1, 1.0, payload)
        result = cache.get(1, 1, 1.0)
        assert result == payload
