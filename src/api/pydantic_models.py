"""Pydantic schemas for the API."""
from typing import List

from pydantic import BaseModel, Field


class Features(BaseModel):
    features: List[float] = Field(..., description="Model input feature vector")


class PredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
