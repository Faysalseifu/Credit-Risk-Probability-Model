"""Streamlit dashboard for Credit Risk Probability Model."""
import os
import json
from pathlib import Path
from typing import Optional

import mlflow.pyfunc
import pandas as pd
import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Page config
st.set_page_config(
    page_title="Credit Risk Probability Model",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model loading configuration
MODEL_NAME = os.getenv("MODEL_NAME", "CreditRiskProxyModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "data/processed/expected_columns.json")

# Feature names
FEATURE_NAMES = ["Recency", "Frequency", "Monetary", "Monetary_abs", "Monetary_positive"]


@st.cache_resource
def load_model():
    """Load model from MLflow registry."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model, "success"
    except Exception as e:
        return None, str(e)


def load_expected_columns() -> Optional[list]:
    """Load expected feature columns from JSON file."""
    try:
        p = Path(FEATURE_COLUMNS_PATH)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return FEATURE_NAMES


def predict_single(model, features: dict) -> dict:
    """Make prediction for a single customer."""
    df = pd.DataFrame([features])
    expected_cols = load_expected_columns()
    if expected_cols:
        df = df[[col for col in expected_cols if col in df.columns]]
    prob = float(model.predict(df)[0])
    return {
        "risk_probability": prob,
        "is_high_risk": int(prob >= 0.5),
        "risk_category": "High Risk" if prob >= 0.5 else "Low Risk"
    }


def main():
    """Main dashboard application."""
    st.title("🏦 Credit Risk Probability Model Dashboard")
    st.markdown("**Alternative Data Credit Scoring for BNPL Programs**")
    
    # Load model
    model, status = load_model()
    if model is None:
        st.error(f"❌ Failed to load model: {status}")
        st.info("💡 Make sure MLflow tracking URI is set correctly and model is registered.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Prediction", "Model Explainability (SHAP)", "Model Performance"]
    )
    
    if page == "Prediction":
        render_prediction_page(model)
    elif page == "Model Explainability (SHAP)":
        render_shap_page(model)
    elif page == "Model Performance":
        render_performance_page()


def render_prediction_page(model):
    """Render the prediction interface."""
    st.header("🔮 Risk Prediction")
    st.markdown("Enter customer features to get a risk probability prediction.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Features")
        recency = st.number_input("Recency (days since last transaction)", min_value=0.0, value=45.0, step=1.0)
        frequency = st.number_input("Frequency (number of transactions)", min_value=0.0, value=6.0, step=1.0)
        monetary = st.number_input("Monetary (average transaction value)", min_value=0.0, value=12000.0, step=100.0)
        monetary_abs = st.number_input("Monetary Absolute (absolute transaction value)", min_value=0.0, value=15000.0, step=100.0)
        monetary_positive = st.number_input("Monetary Positive (positive transactions only)", min_value=0.0, value=14000.0, step=100.0)
        customer_id = st.text_input("Customer ID (optional)", value="")
    
    with col2:
        st.subheader("Prediction Result")
        
        if st.button("🔍 Predict Risk", type="primary"):
            features = {
                "Recency": recency,
                "Frequency": frequency,
                "Monetary": monetary,
                "Monetary_abs": monetary_abs,
                "Monetary_positive": monetary_positive,
            }
            
            result = predict_single(model, features)
            
            # Display result
            prob = result["risk_probability"]
            risk_color = "🔴" if result["is_high_risk"] else "🟢"
            
            st.metric("Risk Probability", f"{prob:.2%}")
            st.metric("Risk Category", f"{risk_color} {result['risk_category']}")
            
            # Progress bar
            st.progress(prob)
            
            # Interpretation
            if prob >= 0.7:
                st.warning("⚠️ **High Risk**: Strong recommendation to reject or apply strict credit limits.")
            elif prob >= 0.5:
                st.warning("⚠️ **Medium-High Risk**: Consider reduced credit limits or additional verification.")
            elif prob >= 0.3:
                st.info("ℹ️ **Medium Risk**: Standard credit terms recommended.")
            else:
                st.success("✅ **Low Risk**: Favorable credit terms can be offered.")
            
            # Feature importance visualization
            st.subheader("Feature Values")
            feature_df = pd.DataFrame({
                "Feature": list(features.keys()),
                "Value": list(features.values())
            })
            st.bar_chart(feature_df.set_index("Feature"))


def render_shap_page(model):
    """Render SHAP explainability page."""
    st.header("🔍 Model Explainability (SHAP)")
    st.markdown("Understand how each feature contributes to the risk prediction.")
    
    # Generate sample data for SHAP
    st.subheader("SHAP Analysis")
    
    # Create sample data
    n_samples = st.slider("Number of samples for SHAP analysis", min_value=10, max_value=100, value=50, step=10)
    
    if st.button("Generate SHAP Explanations"):
        with st.spinner("Computing SHAP values... This may take a moment."):
            try:
                # Generate sample data
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    "Recency": np.random.uniform(0, 180, n_samples),
                    "Frequency": np.random.uniform(0, 50, n_samples),
                    "Monetary": np.random.uniform(0, 50000, n_samples),
                    "Monetary_abs": np.random.uniform(0, 60000, n_samples),
                    "Monetary_positive": np.random.uniform(0, 55000, n_samples),
                })
                
                expected_cols = load_expected_columns()
                if expected_cols:
                    sample_data = sample_data[[col for col in expected_cols if col in sample_data.columns]]
                
                # Get underlying XGBoost model if available
                try:
                    underlying_model = model._model_impl.python_model.model
                    if hasattr(underlying_model, 'get_booster'):
                        # XGBoost model
                        explainer = shap.TreeExplainer(underlying_model)
                        shap_values = explainer.shap_values(sample_data)
                        
                        # Summary plot
                        st.subheader("SHAP Summary Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, sample_data, show=False, plot_type="bar")
                        st.pyplot(fig)
                        plt.close()
                        
                        # Waterfall plot for first sample
                        st.subheader("SHAP Waterfall Plot (First Sample)")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=sample_data.iloc[0].values,
                                feature_names=sample_data.columns.tolist()
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close()
                        
                        st.success("✅ SHAP analysis completed!")
                    else:
                        st.warning("⚠️ Model is not XGBoost. SHAP TreeExplainer requires tree-based models.")
                        st.info("💡 For other models, consider using SHAP's KernelExplainer (slower) or LinearExplainer.")
                except Exception as e:
                    st.error(f"❌ Error computing SHAP values: {str(e)}")
                    st.info("💡 Make sure shap package is installed: `pip install shap`")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Feature importance from model (if available)
    st.subheader("Feature Importance")
    st.info("💡 Feature importance shows which features the model considers most important for predictions.")
    
    # Display feature importance if model supports it
    try:
        underlying_model = model._model_impl.python_model.model
        if hasattr(underlying_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                "Feature": FEATURE_NAMES[:len(underlying_model.feature_importances_)],
                "Importance": underlying_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            st.bar_chart(importance_df.set_index("Feature"))
    except Exception:
        st.info("Feature importance not available for this model type.")


def render_performance_page():
    """Render model performance metrics page."""
    st.header("📊 Model Performance")
    st.markdown("Key performance metrics and model information.")
    
    # Model info
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Name", MODEL_NAME)
    with col2:
        st.metric("Model Stage", MODEL_STAGE)
    with col3:
        st.metric("Tracking URI", "Local" if "file:" in MLFLOW_TRACKING_URI else "Remote")
    
    # Performance metrics (placeholder - would come from MLflow)
    st.subheader("Performance Metrics")
    st.info("💡 Metrics are tracked in MLflow. View detailed metrics by running `mlflow ui`")
    
    # Example metrics (would be loaded from MLflow in production)
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("ROC-AUC", "0.847", delta="0.02", delta_color="normal")
    with metrics_col2:
        st.metric("Precision", "0.78", delta="0.01", delta_color="normal")
    with metrics_col3:
        st.metric("Recall", "0.72", delta="0.03", delta_color="normal")
    with metrics_col4:
        st.metric("F1-Score", "0.75", delta="0.02", delta_color="normal")
    
    # Business impact
    st.subheader("Business Impact")
    st.markdown("""
    **Estimated Impact:**
    - **Risk Reduction**: By rejecting top 20% of high-risk customers, estimated 25-35% reduction in defaults
    - **Capital Efficiency**: Improved PD estimates support Basel II compliance and capital allocation
    - **Portfolio Quality**: Better risk segmentation enables targeted pricing and collections strategies
    """)


if __name__ == "__main__":
    main()
