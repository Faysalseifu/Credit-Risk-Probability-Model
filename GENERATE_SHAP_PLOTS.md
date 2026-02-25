# Generating SHAP Plots for README

This guide helps you generate SHAP plots to include in your README and portfolio.

## Quick Start

```bash
# Set MLflow tracking URI (if models are in notebooks/mlruns)
export MLFLOW_TRACKING_URI="file:./notebooks/mlruns"

# Run SHAP analysis
python -m src.shap_analysis
```

The plots will be saved to `artifacts/shap_plots/`:
- `shap_summary_bar.png` - Feature importance bar chart
- `shap_summary_dot.png` - SHAP value distribution
- `shap_waterfall_sample.png` - Individual prediction explanation
- `shap_feature_importance.png` - Mean absolute SHAP values

## Troubleshooting

### Error: Model not found
If you get an error about the model not being found:

1. **Check MLflow tracking URI:**
   ```bash
   # If your mlruns folder is in notebooks/
   export MLFLOW_TRACKING_URI="file:./notebooks/mlruns"
   
   # Or if it's in the root
   export MLFLOW_TRACKING_URI="file:./mlruns"
   ```

2. **Verify model is registered:**
   ```bash
   mlflow ui --backend-store-uri file:./notebooks/mlruns
   # Open http://localhost:5000 and check Models tab
   ```

3. **Check model name and stage:**
   - Default: `CreditRiskProxyModel` in `Production` stage
   - Override with env vars:
     ```bash
     export MODEL_NAME="CreditRiskProxyModel"
     export MODEL_STAGE="Production"
     ```

### Error: SHAP TreeExplainer requires tree-based models
- Make sure you're using the XGBoost model, not Logistic Regression
- The script will warn you if the model isn't tree-based

### Error: Missing dependencies
```bash
pip install shap matplotlib seaborn
```

## Using SHAP Plots in README

After generating the plots:

1. **Copy to screenshots directory:**
   ```bash
   cp artifacts/shap_plots/shap_waterfall_sample.png artifacts/screenshots/shap_force_plot.png
   ```

2. **Or use summary plot:**
   ```bash
   cp artifacts/shap_plots/shap_summary_bar.png artifacts/screenshots/shap_summary.png
   ```

3. **Update README** to reference the correct image path

## Alternative: Generate from Dashboard

You can also generate SHAP plots directly from the Streamlit dashboard:

1. Run: `streamlit run dashboard.py`
2. Navigate to "Model Explainability (SHAP)" page
3. Click "Generate SHAP Explanations"
4. Take a screenshot of the waterfall plot
5. Save as `artifacts/screenshots/shap_force_plot.png`
