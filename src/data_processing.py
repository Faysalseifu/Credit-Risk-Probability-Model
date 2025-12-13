"""Data loading and preprocessing helpers."""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target column named 'target'."""
    if "target" not in df.columns:
        raise ValueError("DataFrame must include a 'target' column")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/validation split."""
    X, y = preprocess(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
