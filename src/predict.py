"""Inference helpers."""
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
from joblib import load


DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def load_model(path: Path | str = DEFAULT_MODEL_PATH):
    """Load a persisted model; return None if missing."""
    target = Path(path)
    if not target.exists():
        return None
    return load(target)


def _as_model_frame(features: Union[Iterable[float], Mapping[str, object]]) -> Union[np.ndarray, pd.DataFrame]:
    """Convert user input to the right shape for the trained estimator."""
    if isinstance(features, Mapping):
        return pd.DataFrame([features])
    return np.array(list(features), dtype=float).reshape(1, -1)


def predict_proba(model, features: Union[Iterable[float], Mapping[str, object]]) -> float:
    """Return probability of positive class; fall back to neutral."""
    if model is None:
        return 0.5

    frame_or_array = _as_model_frame(features)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame_or_array)[0][1]
        return float(proba)
    prediction = model.predict(frame_or_array)[0]
    return float(prediction)
