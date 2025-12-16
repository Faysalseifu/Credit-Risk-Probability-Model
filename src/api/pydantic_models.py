 """Pydantic schemas for the API."""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Minimal feature schema matching Task 5 training.

    Extend with additional engineered features as you evolve the pipeline.
    """

    # Core RFM features used in training fallback
    Recency: float
    Frequency: float
    Monetary: float
    Monetary_abs: float
    Monetary_positive: float

    customer_id: Optional[str] = Field(default=None, description="Optional customer identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Recency": 45,
                "Frequency": 6,
                "Monetary": 12000.0,
                "Monetary_abs": 15000.0,
                "Monetary_positive": 14000.0,
                "customer_id": "CUST_123",
            }
        }
    }


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = Field(default=None)
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    is_high_risk: int = Field(..., description="0/1 based on probability threshold")
    model_version: str = Field(..., description="Identifier of the loaded model version")


# Backward-compatible generic payload (optional)
class Features(BaseModel):
    features: Dict[str, Any] = Field(..., description="Raw model features keyed by column name")
