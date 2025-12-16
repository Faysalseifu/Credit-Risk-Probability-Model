"""Feature engineering and preprocessing utilities."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse as sp

try:  # Prefer library WoE; fall back to manual if unavailable.
    from xverse.transformer import WOE
except Exception:  # pragma: no cover - optional dependency
    WOE = None


TARGET_COLUMN = "FraudResult"
DATETIME_COLUMN = "TransactionStartTime"
AMOUNT_COLUMN = "Amount"
GROUP_KEY = "CustomerId"


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target column from feature frame."""
    if target not in df.columns:
        raise ValueError(f"DataFrame must include a '{target}' column")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Expand an ISO timestamp column into calendar components."""

    def __init__(self, column: str = DATETIME_COLUMN, drop_original: bool = True):
        self.column = column
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # noqa: D401
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input frame")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        dt = pd.to_datetime(df[self.column], errors="coerce")
        df["transaction_hour"] = dt.dt.hour
        df["transaction_day"] = dt.dt.day
        df["transaction_month"] = dt.dt.month
        df["transaction_year"] = dt.dt.year
        if self.drop_original:
            df = df.drop(columns=[self.column])
        return df


class AggregateByGroup(BaseEstimator, TransformerMixin):
    """Customer-level aggregates for transaction amounts."""

    def __init__(self, group_key: str = GROUP_KEY, amount_column: str = AMOUNT_COLUMN):
        self.group_key = group_key
        self.amount_column = amount_column
        self._agg_frame: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # noqa: D401
        missing = [c for c in (self.group_key, self.amount_column) if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        grouped = X.groupby(self.group_key)[self.amount_column].agg(["sum", "mean", "count", "std"]).reset_index()
        grouped = grouped.rename(
            columns={
                "sum": "total_amount_sum",
                "mean": "average_amount",
                "count": "transaction_count",
                "std": "amount_std",
            }
        )
        grouped["amount_std"] = grouped["amount_std"].fillna(0.0)
        self._agg_frame = grouped
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._agg_frame is None:
            raise RuntimeError("Transformer has not been fitted")
        df = X.copy()
        df = df.merge(self._agg_frame, on=self.group_key, how="left")
        for col in ["total_amount_sum", "average_amount", "transaction_count", "amount_std"]:
            df[col] = df[col].fillna(0.0)
        return df


class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """Apply Weight of Evidence encoding to categorical columns."""

    def __init__(self, columns: Optional[Iterable[str]] = None, suffix: str = "_woe", epsilon: float = 1e-6):
        self.columns = list(columns) if columns is not None else None
        self.suffix = suffix
        self.epsilon = epsilon
        self.encoder_: Optional[WOE] = None
        self.cols_: List[str] = []
        self._woe_maps: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):  # noqa: D401
        if y.nunique() > 2:
            raise ValueError("Weight of Evidence requires binary target")

        self.cols_ = self.columns or [c for c in X.columns if X[c].dtype == object]
        if not self.cols_:
            return self

        target = pd.Series(y, name=y.name or "target")

        # Manual robust WoE for categoricals
        total_pos = float((target == 1).sum()) + self.epsilon
        total_neg = float((target == 0).sum()) + self.epsilon
        for col in self.cols_:
            stats = pd.concat([X[col], target], axis=1)
            grouped = stats.groupby(col)[target.name].agg(["sum", "count"])
            grouped["non_event"] = grouped["count"] - grouped["sum"]
            grouped["event_rate"] = (grouped["sum"] + self.epsilon) / total_pos
            grouped["non_event_rate"] = (grouped["non_event"] + self.epsilon) / total_neg
            grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
            self._woe_maps[col] = grouped["woe"].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cols_:
            return X

        df = X.copy()
        encoded = pd.DataFrame(index=df.index)
        for col in self.cols_:
            mapping = self._woe_maps.get(col, {})
            encoded[f"{col}{self.suffix}"] = df[col].map(mapping).fillna(0.0)

        df = df.drop(columns=self.cols_)
        df = pd.concat([df, encoded], axis=1)
        return df


def build_preprocessing_pipeline(
    datetime_column: str = DATETIME_COLUMN,
    group_key: str = GROUP_KEY,
    amount_column: str = AMOUNT_COLUMN,
    woe_columns: Optional[Iterable[str]] = None,
    scaler: str = "standard",
    categorical_encoding: str = "onehot",
) -> Pipeline:
    """Create a reusable preprocessing pipeline with feature engineering.

    Parameters
    - datetime_column: timestamp column to expand into calendar parts
    - group_key: key used for customer-level aggregations
    - amount_column: numeric amount column to aggregate by group
    - woe_columns: which categorical columns to WoE-encode (None => all object)
    - scaler: one of {"standard", "minmax"}
    - categorical_encoding: one of {"onehot", "ordinal"}
    """

    scaler_step = StandardScaler() if scaler == "standard" else MinMaxScaler() if scaler == "minmax" else StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_step),
        ]
    )

    # Choose categorical encoder
    if categorical_encoding == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore")
    elif categorical_encoding == "ordinal":
        from sklearn.preprocessing import OrdinalEncoder

        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError("categorical_encoding must be 'onehot' or 'ordinal'")

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", cat_encoder),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, make_column_selector(dtype_include=["object", "category"])),
        ]
    )

    return Pipeline(
        steps=[
            ("datetime", DateTimeFeatureExtractor(column=datetime_column, drop_original=True)),
            ("aggregate", AggregateByGroup(group_key=group_key, amount_column=amount_column)),
            ("woe", WeightOfEvidenceEncoder(columns=woe_columns)),
            ("encode", column_transformer),
        ]
    )


def compute_woe_iv(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    target: str = TARGET_COLUMN,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Compute Weight of Evidence (WoE) and Information Value (IV) for categorical columns.

    Returns a tuple of (detail_frame, woe_maps, iv_per_column).

    - detail_frame has rows per (column, category) with event_rate, non_event_rate, woe, iv_component.
    - woe_maps is a mapping of column -> {category -> woe}.
    - iv_per_column is a mapping of column -> total IV.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    y = df[target]
    if y.nunique() > 2:
        raise ValueError("Information Value requires a binary target")

    cat_cols = list(columns) if columns is not None else [c for c in df.columns if df[c].dtype == object and c != target]
    total_pos = float((y == 1).sum()) + epsilon
    total_neg = float((y == 0).sum()) + epsilon

    records = []
    woe_maps: dict = {}
    iv_per_col: dict = {}
    for col in cat_cols:
        stats = pd.concat([df[col], y], axis=1)
        grouped = stats.groupby(col)[y.name].agg(["sum", "count"]).rename(columns={"sum": "events", "count": "total"})
        grouped["non_events"] = grouped["total"] - grouped["events"]
        grouped["event_rate"] = (grouped["events"] + epsilon) / total_pos
        grouped["non_event_rate"] = (grouped["non_events"] + epsilon) / total_neg
        grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
        grouped["iv_component"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]

        woe_maps[col] = grouped["woe"].to_dict()
        iv_per_col[col] = float(grouped["iv_component"].sum())

        for category, row in grouped.iterrows():
            records.append(
                {
                    "column": col,
                    "category": category,
                    "events": int(row["events"]),
                    "non_events": int(row["non_events"]),
                    "event_rate": float(row["event_rate"]),
                    "non_event_rate": float(row["non_event_rate"]),
                    "woe": float(row["woe"]),
                    "iv_component": float(row["iv_component"]),
                }
            )

    detail = pd.DataFrame.from_records(records)
    return detail, woe_maps, iv_per_col


def train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/validation split."""
    X, y = split_features_target(df, target=target)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def transform_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    woe_columns: Optional[Iterable[str]] = None,
) -> Tuple[sp.spmatrix | np.ndarray, pd.Series, List[str]]:
    """Fit preprocessing on provided data and return transformed matrix, target and feature names.

    This is useful to create a model-ready dataset as an artifact for inspection or offline modeling.
    """
    X, y = split_features_target(df, target=target)
    pipe = build_preprocessing_pipeline(woe_columns=woe_columns)
    X_t = pipe.fit_transform(X, y)

    # Try to extract feature names from the ColumnTransformer step.
    try:
        encode = pipe.named_steps["encode"]
        feature_names = list(encode.get_feature_names_out())
    except Exception:  # pragma: no cover - feature names are best-effort
        feature_names = [f"f{i}" for i in range(X_t.shape[1])]
    return X_t, y, feature_names


def process_raw_data(
    input_path: str = "data/raw/data.csv",
    out_dir: str = "data/processed",
    target: str = TARGET_COLUMN,
    woe_columns: Optional[Iterable[str]] = None,
) -> dict:
    """Process raw CSV into model-ready artifacts.

    Saves the following into out_dir:
      - features.npz (sparse) or features.npy (dense)
      - target.csv
      - feature_names.txt
    Returns a mapping with file paths.
    """
    df = load_raw(input_path)
    X_t, y, feature_names = transform_features(df, target=target, woe_columns=woe_columns)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    # Save features
    if sp.issparse(X_t):
        fpath = out / "features.npz"
        sp.save_npz(fpath, X_t)
        paths["features"] = str(fpath)
    else:
        fpath = out / "features.npy"
        np.save(fpath, X_t)
        paths["features"] = str(fpath)

    # Save target
    ypath = out / "target.csv"
    pd.DataFrame({target: y}).to_csv(ypass := ypath, index=False)
    paths["target"] = str(ypass)

    # Save feature names
    npath = out / "feature_names.txt"
    with npath.open("w", encoding="utf-8") as fh:
        for name in feature_names:
            fh.write(str(name))
            fh.write("\n")
    paths["feature_names"] = str(npath)

    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process raw credit risk data into model-ready features")
    parser.add_argument("--input", default="data/raw/data.csv", help="Path to raw CSV")
    parser.add_argument("--out", default="data/processed", help="Output directory for processed artifacts")
    parser.add_argument(
        "--woe-cols",
        nargs="*",
        default=None,
        help="Optional list of categorical columns to WoE encode (defaults to all object dtype)",
    )
    parser.add_argument("--target", default=TARGET_COLUMN, help="Target column name")
    return parser.parse_args()


def _main_cli() -> None:  # pragma: no cover - thin CLI wrapper
    args = _parse_args()
    paths = process_raw_data(input_path=args.input, out_dir=args.out, target=args.target, woe_columns=args.woe_cols)
    print("Processed artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":  # pragma: no cover
    _main_cli()
