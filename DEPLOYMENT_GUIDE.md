# Deployment Guide

Quick guide to deploy your Credit Risk Model for portfolio showcase.

## 🚀 Option 1: Streamlit Community Cloud (Recommended - Fastest)

**Time: 15-30 minutes**

1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "chore: portfolio polish - README, screenshots, deployment ready"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `Credit-Risk-Probability-Model`
   - Main file path: `dashboard.py`
   - Branch: `main`
   - Click "Deploy"

3. **Configure Environment Variables** (if needed):
   - In Streamlit Cloud settings, add:
     - `MLFLOW_TRACKING_URI`: `file:./notebooks/mlruns` (or your path)
     - `MODEL_NAME`: `CreditRiskProxyModel`
     - `MODEL_STAGE`: `Production`

4. **Update README:**
   - Copy your Streamlit Cloud URL (e.g., `https://your-app.streamlit.app`)
   - Update line 68 in README.md:
     ```markdown
     - **Streamlit Dashboard**: [🔗 Live Demo](https://your-app.streamlit.app)
     ```

## 🌐 Option 2: Render.com (More Control)

**Time: 30-45 minutes**

1. **Create `render.yaml`** (optional, for infrastructure as code):
   ```yaml
   services:
     - type: web
       name: credit-risk-dashboard
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
       envVars:
         - key: MLFLOW_TRACKING_URI
           value: file:./notebooks/mlruns
   ```

2. **Deploy:**
   - Go to https://render.com
   - Sign in with GitHub
   - New → Web Service
   - Connect your repository
   - Settings:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - Add environment variables (same as Streamlit Cloud)
   - Deploy

## 🐳 Option 3: Docker + Cloud Platform

**Time: 45-60 minutes**

### For API (FastAPI):

1. **Deploy to Railway/Render/Fly.io:**
   ```bash
   # Railway
   railway login
   railway init
   railway up
   
   # Or Render
   # Use docker-compose.yml, Render will auto-detect it
   ```

2. **Update README** with API endpoint URL

### For Dashboard:

Same as Option 1 or 2, but using Docker:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

## ✅ Post-Deployment Checklist

- [ ] Test dashboard loads correctly
- [ ] Test prediction endpoint works
- [ ] Test SHAP visualizations render
- [ ] Update README with live demo links
- [ ] Test links work from GitHub README
- [ ] Share on LinkedIn with screenshots

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `MLFLOW_TRACKING_URI` points to correct path
- Upload model artifacts to cloud storage (S3/GCS) if needed
- Or use MLflow Model Registry (cloud)

### Dependencies Missing
- Check `requirements.txt` includes all packages
- Test locally first: `pip install -r requirements.txt`

### Port Issues
- Streamlit: Use `--server.port $PORT` (Render/Railway provide PORT env var)
- FastAPI: Use `uvicorn app:app --host 0.0.0.0 --port $PORT`

## 📝 Quick Commands

```bash
# Test locally first
streamlit run dashboard.py

# Test API locally
docker compose up
# Or
uvicorn src.api.main:app --reload

# Generate SHAP plots
python -m src.shap_analysis

# Take screenshots
python scripts/generate_screenshots.py --open-dashboard
```
