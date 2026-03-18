"""
schemas.py – Pydantic v2 request / response models for the Pulsecast API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """Request body for the POST /forecast endpoint."""

    route_id: int = Field(
        ...,
        description="TLC zone / route identifier (PULocationID).",
        ge=1,
    )
    horizon: int = Field(
        ...,
        description="Forecast horizon in days (1–7).",
        ge=1,
        le=7,
    )

    @field_validator("horizon")
    @classmethod
    def horizon_within_bounds(cls, v: int) -> int:
        if not 1 <= v <= 7:
            raise ValueError("horizon must be between 1 and 7 days inclusive.")
        return v


class ForecastResponse(BaseModel):
    """Probabilistic demand forecast (p10 / p50 / p90) per step."""

    route_id: int = Field(..., description="Echo of the requested route_id.")
    horizon: int = Field(..., description="Echo of the requested horizon (days).")
    p10: list[float] = Field(
        ...,
        description="10th-percentile forecast for each hour in the horizon.",
    )
    p50: list[float] = Field(
        ...,
        description="Median forecast for each hour in the horizon.",
    )
    p90: list[float] = Field(
        ...,
        description="90th-percentile forecast for each hour in the horizon.",
    )

    model_config = {"json_schema_extra": {"example": {
        "route_id": 132,
        "horizon": 1,
        "p10": [42.1],
        "p50": [58.3],
        "p90": [75.6],
    }}}
