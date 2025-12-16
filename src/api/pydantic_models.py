"""Pydantic schemas for the API."""
from typing import Any, Dict

from pydantic import BaseModel, Field


class Features(BaseModel):
    features: Dict[str, Any] = Field(..., description="Raw model features keyed by column name")


class PredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
