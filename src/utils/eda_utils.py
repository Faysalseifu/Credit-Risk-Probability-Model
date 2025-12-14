"""Lightweight EDA helper functions used in notebooks."""
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a CSV file from disk with a friendly error if missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path)


def quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary with dtype, non-null count, and unique values."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_nulls": df.notnull().sum(),
            "null_pct": df.isnull().mean().mul(100).round(2),
            "unique": df.nunique(dropna=False),
        }
    )


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Describe numeric columns and append missing percentages."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    summary = num.describe().T
    summary["missing_pct"] = 100 - num.notnull().mean().mul(100)
    return summary


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for plotting functions; install via `pip install matplotlib`"
        ) from exc
    return plt


def plot_num_distributions(df: pd.DataFrame):
    """Plot histograms for numeric columns (returns matplotlib axes)."""
    plt = _require_matplotlib()
    num = df.select_dtypes(include="number")
    if num.empty:
        print("No numeric columns to plot.")
        return None
    axes = num.hist(bins=30, figsize=(12, 10))
    plt.tight_layout()
    return axes


def plot_cat_distributions(df: pd.DataFrame, top_n: int = 20):
    """Plot bar charts for categorical columns showing top frequencies."""
    plt = _require_matplotlib()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        print("No categorical columns to plot.")
        return None

    n = len(cat_cols)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)

    for ax, col in zip(axes.flat, cat_cols):
        counts = df[col].value_counts(dropna=False).head(top_n)
        counts.plot(kind="bar", ax=ax, title=col)
        ax.set_ylabel("count")

    for ax in axes.flat[len(cat_cols) :]:
        ax.remove()

    plt.tight_layout()
    return axes


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot a simple correlation heatmap for numeric columns."""
    plt = _require_matplotlib()
    num = df.select_dtypes(include="number")
    if num.empty:
        print("No numeric columns for correlation heatmap.")
        return None

    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation heatmap")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return ax


def boxplot_outliers(df: pd.DataFrame, columns: Iterable[str]):
    """Plot boxplots for selected columns to inspect outliers."""
    plt = _require_matplotlib()
    cols = [c for c in columns if c in df.columns]
    if not cols:
        print("No matching columns to plot.")
        return None

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        df[[col]].boxplot(ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    return axes
