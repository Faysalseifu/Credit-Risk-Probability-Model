"""Inference helpers."""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from joblib import load


DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def load_model(path: Path | str = DEFAULT_MODEL_PATH):
    """Load a persisted model; return None if missing."""
    target = Path(path)
    if not target.exists():
        return None
    return load(target)


def predict_proba(model, features: Iterable[float]) -> float:
    """Return probability of positive class; fall back to neutral."""
    if model is None:
        return 0.5
    array = np.array(list(features), dtype=float).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(array)[0][1]
        return float(proba)
    prediction = model.predict(array)[0]
    return float(prediction)
