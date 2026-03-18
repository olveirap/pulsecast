"""
tests/test_schemas.py – Pydantic schema validation tests.

Covers:
  - ForecastRequest: valid inputs, horizon bounds, route_id constraints.
  - ForecastResponse: well-formed response round-trips.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pulsecast.serving.schemas import ForecastRequest, ForecastResponse

# ---------------------------------------------------------------------------
# ForecastRequest
# ---------------------------------------------------------------------------


class TestForecastRequestValid:
    def test_minimal_valid(self):
        req = ForecastRequest(route_id=1, horizon=1)
        assert req.route_id == 1
        assert req.horizon == 1

    def test_max_horizon(self):
        req = ForecastRequest(route_id=132, horizon=7)
        assert req.horizon == 7

    def test_large_route_id(self):
        req = ForecastRequest(route_id=999, horizon=3)
        assert req.route_id == 999


class TestForecastRequestHorizonConstraints:
    def test_horizon_zero_is_invalid(self):
        with pytest.raises(ValidationError):
            ForecastRequest(route_id=1, horizon=0)

    def test_horizon_eight_is_invalid(self):
        with pytest.raises(ValidationError):
            ForecastRequest(route_id=1, horizon=8)

    def test_horizon_negative_is_invalid(self):
        with pytest.raises(ValidationError):
            ForecastRequest(route_id=1, horizon=-1)

    @pytest.mark.parametrize("h", [1, 2, 3, 4, 5, 6, 7])
    def test_all_valid_horizons_accepted(self, h: int):
        req = ForecastRequest(route_id=1, horizon=h)
        assert req.horizon == h


class TestForecastRequestRouteIdConstraints:
    def test_route_id_zero_is_invalid(self):
        with pytest.raises(ValidationError):
            ForecastRequest(route_id=0, horizon=1)

    def test_route_id_negative_is_invalid(self):
        with pytest.raises(ValidationError):
            ForecastRequest(route_id=-5, horizon=1)

    def test_route_id_one_is_valid(self):
        req = ForecastRequest(route_id=1, horizon=1)
        assert req.route_id == 1


class TestForecastRequestFieldValidatorMessage:
    def test_horizon_validator_error_message(self):
        """Validator error must mention the horizon constraint."""
        with pytest.raises(ValidationError) as exc_info:
            ForecastRequest(route_id=1, horizon=10)
        errors = exc_info.value.errors()
        assert any("horizon" in str(e).lower() for e in errors)


# ---------------------------------------------------------------------------
# ForecastResponse
# ---------------------------------------------------------------------------


class TestForecastResponseValid:
    def test_basic_construction(self):
        resp = ForecastResponse(
            route_id=1,
            horizon=1,
            p10=[10.0],
            p50=[20.0],
            p90=[30.0],
        )
        assert resp.route_id == 1
        assert resp.horizon == 1
        assert resp.p10 == [10.0]
        assert resp.p50 == [20.0]
        assert resp.p90 == [30.0]

    def test_multi_step_forecast(self):
        n = 24
        resp = ForecastResponse(
            route_id=5,
            horizon=1,
            p10=list(range(n)),
            p50=list(range(n, 2 * n)),
            p90=list(range(2 * n, 3 * n)),
        )
        assert len(resp.p10) == n
        assert len(resp.p50) == n
        assert len(resp.p90) == n

    def test_json_roundtrip(self):
        resp = ForecastResponse(
            route_id=132,
            horizon=2,
            p10=[1.1, 2.2],
            p50=[3.3, 4.4],
            p90=[5.5, 6.6],
        )
        data = resp.model_dump()
        assert data["route_id"] == 132
        assert data["p10"] == [1.1, 2.2]

    def test_model_config_example_is_valid(self):
        """The embedded JSON schema example must be a valid ForecastResponse."""
        example = ForecastResponse.model_config["json_schema_extra"]["example"]
        resp = ForecastResponse(**example)
        assert resp.route_id == 132


class TestForecastResponseInvalid:
    def test_missing_p10_raises(self):
        with pytest.raises(ValidationError):
            ForecastResponse(route_id=1, horizon=1, p50=[1.0], p90=[2.0])

    def test_missing_route_id_raises(self):
        with pytest.raises(ValidationError):
            ForecastResponse(horizon=1, p10=[1.0], p50=[2.0], p90=[3.0])
