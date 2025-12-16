"""FastAPI application that serves predictions from MLflow Registry."""
import os
import json
from pathlib import Path
from typing import Optional, List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CustomerFeatures, PredictionResponse, Features


app = FastAPI(
    title="Credit Risk Probability Model",
    version="1.0.0",
    description="Serves PD scores from the best model in MLflow Registry",
)

MODEL_NAME = os.getenv("MODEL_NAME", "CreditRiskProxyModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "")


@app.on_event("startup")
def load_model() -> None:
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        app.state.model = model
        # Use run_id prefix as a lightweight version tag
        app.state.model_version = getattr(model.metadata, "run_id", "unknown")[:8]
        # Load expected feature columns, if provided
        cols_path = FEATURE_COLUMNS_PATH.strip()
        if cols_path:
            p = Path(cols_path)
            if p.exists():
                with p.open("r", encoding="utf-8") as fh:
                    app.state.expected_columns = json.load(fh)
            else:
                app.state.expected_columns = None
        else:
            app.state.expected_columns = None
    except Exception as e:  # pragma: no cover - runtime dependent
        raise RuntimeError(f"Failed to load model from MLflow: {e}")


@app.get("/")
def health() -> dict:
    return {"status": "healthy", "model_version": getattr(app.state, "model_version", "n/a")}


def _predict_df(df: pd.DataFrame) -> float:
    if not hasattr(app.state, "model"):
        raise RuntimeError("Model is not loaded")
    exp_cols: Optional[List[str]] = getattr(app.state, "expected_columns", None)
    # Apply expected column order if available
    if exp_cols:
        missing = [c for c in exp_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        df = df[exp_cols]
    # pyfunc model returns array-like or Series
    proba = app.state.model.predict(df)
    # Ensure scalar float
    return float(proba[0])


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures) -> PredictionResponse:
    try:
        input_df = pd.DataFrame([
            {
                "Recency": features.Recency,
                "Frequency": features.Frequency,
                "Monetary": features.Monetary,
                "Monetary_abs": features.Monetary_abs,
                "Monetary_positive": features.Monetary_positive,
            }
        ])
        prob = _predict_df(input_df)
        pred = int(prob >= THRESHOLD)
        return PredictionResponse(
            customer_id=features.customer_id,
            risk_probability=prob,
            is_high_risk=pred,
            model_version=getattr(app.state, "model_version", "n/a"),
        )
    except Exception as e:  # pragma: no cover - runtime dependent
        raise HTTPException(status_code=500, detail=str(e))


# Backward compatible endpoint for generic feature dicts
@app.post("/predict_raw", response_model=PredictionResponse)
def predict_raw(payload: Features) -> PredictionResponse:
    try:
        df = pd.DataFrame([payload.features])
        prob = _predict_df(df)
        pred = int(prob >= THRESHOLD)
        return PredictionResponse(
            customer_id=None,
            risk_probability=prob,
            is_high_risk=pred,
            model_version=getattr(app.state, "model_version", "n/a"),
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))
