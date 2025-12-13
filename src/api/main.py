"""FastAPI application entrypoint."""
from fastapi import FastAPI

from src.api.pydantic_models import Features, PredictionResponse
from src.predict import load_model, predict_proba

app = FastAPI(title="Credit Risk Probability Model", version="0.1.0")
model = load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: Features) -> PredictionResponse:
    probability = predict_proba(model, payload.features)
    return PredictionResponse(probability=probability)
