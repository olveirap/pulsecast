"""
cache.py – Redis cache layer for the Pulsecast serving API.

Delay index values are bucketed to the nearest 0.5 before being used as
part of the cache key, so that near-identical congestion conditions hit the
same cached forecast.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes default


def _bucket_delay(delay_index: float, bucket_size: float = 0.5) -> float:
    """Round *delay_index* to the nearest *bucket_size*."""
    return round(round(delay_index / bucket_size) * bucket_size, 6)


def _make_key(route_id: int, horizon: int, delay_index: float) -> str:
    bucketed = _bucket_delay(delay_index)
    return f"forecast:{route_id}:{horizon}:{bucketed}"


class ForecastCache:
    """
    Thin Redis-backed cache for probabilistic forecasts.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (default from ``REDIS_URL`` env var).
    ttl : int
        Cache entry TTL in seconds (default from ``CACHE_TTL_SECONDS`` env var).
    """

    def __init__(
        self,
        redis_url: str = _REDIS_URL,
        ttl: int = _TTL_SECONDS,
    ) -> None:
        try:
            import redis

            self._client = redis.from_url(redis_url, decode_responses=True)
        except ImportError as exc:
            raise ImportError(
                "redis-py is required. Install it with: pip install redis"
            ) from exc
        self._ttl = ttl

    def get(
        self, route_id: int, horizon: int, delay_index: float
    ) -> dict[str, Any] | None:
        """
        Retrieve a cached forecast, or *None* on cache miss.
        """
        key = _make_key(route_id, horizon, delay_index)
        raw = self._client.get(key)
        if raw is None:
            logger.debug("Cache MISS for key=%s", key)
            return None
        logger.debug("Cache HIT for key=%s", key)
        return json.loads(str(raw))

    def set(
        self,
        route_id: int,
        horizon: int,
        delay_index: float,
        payload: dict[str, Any],
    ) -> None:
        """
        Store a forecast payload, expiring after *ttl* seconds.
        """
        key = _make_key(route_id, horizon, delay_index)
        self._client.set(key, json.dumps(payload), ex=self._ttl)
        logger.debug("Cache SET for key=%s (ttl=%ds)", key, self._ttl)
