# Scripts Directory

Helper scripts for project maintenance and documentation.

## generate_screenshots.py

Helps generate screenshots for the README.

**Usage:**
```bash
# Show instructions
python scripts/generate_screenshots.py

# Open dashboard automatically
python scripts/generate_screenshots.py --open-dashboard

# Open API docs automatically
python scripts/generate_screenshots.py --open-api
```

**Manual Steps:**
1. Run the dashboard: `streamlit run dashboard.py`
2. Navigate to http://localhost:8501
3. Take screenshots of:
   - Main prediction page → `artifacts/screenshots/dashboard_home.png`
   - SHAP explainability page → `artifacts/screenshots/shap_force_plot.png`
4. Run API: `docker compose up`
5. Open http://localhost:8000/docs
6. Take screenshot → `artifacts/screenshots/api_swagger.png`
