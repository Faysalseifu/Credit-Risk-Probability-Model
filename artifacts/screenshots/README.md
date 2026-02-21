# Screenshots Directory

Place your project screenshots here:

1. `dashboard_home.png` - Streamlit dashboard home page
2. `shap_force_plot.png` - SHAP waterfall/force plot example
3. `api_swagger.png` - FastAPI Swagger documentation page

To generate screenshots:
1. Run the Streamlit dashboard: `streamlit run dashboard.py`
2. Take screenshots of the dashboard pages
3. Run the API: `docker compose up`
4. Take a screenshot of `http://localhost:8000/docs`
5. Generate SHAP plots: `python -m src.shap_analysis`
6. Use SHAP plots from `artifacts/shap_plots/` for the SHAP screenshot
