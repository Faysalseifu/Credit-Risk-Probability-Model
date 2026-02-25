"""SHAP analysis script for model explainability."""
import os
import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "CreditRiskProxyModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
# Default to notebooks/mlruns if it exists, otherwise ./mlruns
_default_uri = "file:./notebooks/mlruns" if Path("notebooks/mlruns").exists() else "file:./mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _default_uri)
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "data/processed/expected_columns.json")
OUTPUT_DIR = Path("artifacts/shap_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(model_uri)


def load_expected_columns() -> list:
    """Load expected feature columns."""
    try:
        p = Path(FEATURE_COLUMNS_PATH)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return ["Recency", "Frequency", "Monetary", "Monetary_abs", "Monetary_positive"]


def generate_shap_plots(model, sample_data: pd.DataFrame, n_samples: int = 100):
    """Generate SHAP plots for model explainability."""
    try:
        # Extract underlying model from MLflow wrapper
        if hasattr(model._model_impl, 'python_model'):
            underlying_model = model._model_impl.python_model.model
        elif hasattr(model._model_impl, 'xgb_model'):
            underlying_model = model._model_impl.xgb_model
        else:
            underlying_model = model._model_impl
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(underlying_model)
        shap_values = explainer.shap_values(sample_data)
        
        # 1. Summary plot (bar)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_data, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {OUTPUT_DIR / 'shap_summary_bar.png'}")
        
        # 2. Summary plot (dot)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, sample_data, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_summary_dot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {OUTPUT_DIR / 'shap_summary_dot.png'}")
        
        # 3. Waterfall plot for first sample
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=sample_data.iloc[0].values,
                feature_names=sample_data.columns.tolist()
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {OUTPUT_DIR / 'shap_waterfall_sample.png'}")
        
        # 4. Feature importance plot
        feature_importance = pd.DataFrame({
            "Feature": sample_data.columns,
            "Importance": np.abs(shap_values).mean(0)
        }).sort_values("Importance", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x="Importance", y="Feature", palette="viridis")
        plt.title("SHAP Feature Importance (Mean |SHAP Value|)")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {OUTPUT_DIR / 'shap_feature_importance.png'}")
        
        # Save feature importance as CSV
        feature_importance.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
        print(f"✅ Saved: {OUTPUT_DIR / 'shap_feature_importance.csv'}")
        
        print(f"\n🎉 SHAP analysis complete! Plots saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error generating SHAP plots: {e}")
        raise


def main():
    """Main function to run SHAP analysis."""
    print("Loading model from MLflow...")
    model = load_model()
    print("✅ Model loaded successfully!")
    
    # Load or generate sample data
    expected_cols = load_expected_columns()
    
    # Generate synthetic sample data (in production, use test set)
    print("Generating sample data for SHAP analysis...")
    np.random.seed(42)
    sample_data = pd.DataFrame({
        "Recency": np.random.uniform(0, 180, 100),
        "Frequency": np.random.uniform(0, 50, 100),
        "Monetary": np.random.uniform(0, 50000, 100),
        "Monetary_abs": np.random.uniform(0, 60000, 100),
        "Monetary_positive": np.random.uniform(0, 55000, 100),
    })
    
    # Ensure correct column order
    if expected_cols:
        sample_data = sample_data[[col for col in expected_cols if col in sample_data.columns]]
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Features: {list(sample_data.columns)}")
    
    # Generate SHAP plots
    generate_shap_plots(model, sample_data)


if __name__ == "__main__":
    main()
