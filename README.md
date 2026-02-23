# Credit Risk Probability Model (Alternative Data)

[![CI](https://github.com/YOUR_USERNAME/Credit-Risk-Probability-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/Credit-Risk-Probability-Model/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-Latest-0194E2.svg)

**Production-grade credit risk scoring for BNPL programs using alternative data and proxy-based risk modeling.**

End-to-end pipeline for building, training, and deploying a credit-risk scoring service for Bati Bank's Buy-Now-Pay-Later (BNPL) program using alternative data and proxy-based risk modeling.

## 🎯 Business Problem

**Challenge**: Bati Bank's BNPL program lacks direct default labels, making traditional supervised learning impossible. Without labeled data, the bank cannot:
- Assess credit risk for new customers
- Comply with Basel II capital requirements (PD estimation)
- Optimize portfolio quality and reduce defaults

**Solution**: Build a proxy-based risk model using RFM (Recency-Frequency-Monetary) analysis and alternative transaction data to predict customer risk without requiring historical defaults.

## 💡 Solution Overview

1. **Proxy Target Engineering**: RFM features + K-means clustering to create risk labels
2. **Model Training**: XGBoost gradient boosting (selected over Logistic Regression)
3. **Explainability**: SHAP analysis for regulatory compliance and interpretability
4. **Deployment**: FastAPI REST API + Streamlit dashboard for real-time predictions
5. **MLOps**: MLflow tracking, Docker containerization, CI/CD pipeline

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.9996** (XGBoost) vs 0.9679 (Logistic Regression) |
| **Precision** | **0.9957** |
| **Recall** | **0.9708** |
| **F1-Score** | **0.9831** |
| **Business Impact** | Estimated **25-35% reduction in high-risk approvals** |
| **Compliance** | Basel II-ready with interpretable features & SHAP explanations |

## 📸 Screenshots

### Dashboard Home Page
![Dashboard](artifacts/screenshots/dashboard_home.png)
*Interactive Streamlit dashboard for risk predictions and model explainability*

### SHAP Force Plot Example
![SHAP Force Plot](artifacts/screenshots/shap_force_plot.png)
*SHAP waterfall plot showing feature contributions for individual predictions*

### API Swagger Documentation
![API Swagger](artifacts/screenshots/api_swagger.png)
*FastAPI auto-generated documentation with `/predict` endpoint*

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Credit-Risk-Probability-Model.git
   cd Credit-Risk-Probability-Model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests**
   ```bash
   pytest -q
   ```

4. **Launch Streamlit Dashboard** (recommended)
   ```bash
   streamlit run dashboard.py
   ```
   Dashboard will be available at `http://localhost:8501`

5. **Run API locally** (alternative)
   ```bash
   docker compose up
   ```
   API will be available at `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/`

## 📊 Live Demo

- **Streamlit Dashboard**: [🔗 Deploy your dashboard and add link here]
- **API Endpoint**: [🔗 Deploy your API and add link here]

## 🏗️ Project Structure

```
credit-risk-model/
├── data/                        # Git-ignored raw & processed data
│   ├── raw/                     # Raw data files (e.g., data.csv)
│   └── processed/               # Cleaned / feature-engineered data
├── notebooks/
│   ├── eda.ipynb                # Exploratory data analysis
│   ├── Model_training.ipynb     # Model training experiments
│   └── target_engineering.ipynb # Proxy target creation
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering utilities
│   ├── train.py                 # Model training script (CLI)
│   ├── predict.py               # Batch inference script (CLI)
│   ├── shap_analysis.py         # SHAP explainability analysis
│   └── api/
│       ├── main.py              # FastAPI app exposing prediction endpoint
│       └── pydantic_models.py   # Request/response schemas
├── tests/
│   └── test_data_processing.py  # Unit tests
├── artifacts/
│   └── shap_plots/              # Generated SHAP visualizations
├── dashboard.py                 # Streamlit dashboard application
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Local orchestration (API + model)
├── requirements.txt             # Python dependencies
└── README.md                    # Project docs
```

## 🔧 Usage

### Model Training

Train models with MLflow experiment tracking:

```bash
python -m src.train \
    --raw-path data/raw/data.csv \
    --model-out artifacts/best_model.pkl
```

View experiments:
```bash
mlflow ui
# Opens MLflow UI at http://127.0.0.1:5000
```

### Generate SHAP Explanations

Create SHAP plots for model explainability:

```bash
python -m src.shap_analysis
```

Outputs saved to `artifacts/shap_plots/`:
- `shap_summary_bar.png` - Feature importance bar chart
- `shap_summary_dot.png` - SHAP value distribution
- `shap_waterfall_sample.png` - Individual prediction explanation
- `shap_feature_importance.png` - Mean absolute SHAP values

### API Usage

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_123",
    "Recency": 45,
    "Frequency": 6,
    "Monetary": 12000.0,
    "Monetary_abs": 15000.0,
    "Monetary_positive": 14000.0
  }'
```

Response:
```json
{
  "customer_id": "CUST_123",
  "risk_probability": 0.37,
  "is_high_risk": 0,
  "model_version": "a5395208"
}
```

#### Batch Prediction

Use the batch inference script:
```bash
python -m src.predict --input data/raw/batch.csv --output predictions.csv
```

## 🎓 Model Architecture

### Proxy Target Engineering

Since BNPL data lacks direct default labels, we construct a proxy target using **RFM (Recency-Frequency-Monetary) analysis** and **K-means clustering**:

1. **RFM Features**: Compute customer-level metrics
   - Recency: Days since last transaction
   - Frequency: Number of transactions
   - Monetary: Transaction value aggregates

2. **Clustering**: Apply K-means (k=3) to segment customers
3. **Label Assignment**: Map least-engaged cluster to `is_high_risk=1`

### Feature Engineering

- **Temporal Features**: Rolling windows, transaction velocity, inter-event times
- **Encodings**: Target encoding (cross-fold), one-hot for low-cardinality
- **WoE/IV**: Monotonic binning for interpretability
- **Risk Flags**: Missingness indicators, extreme value flags

### Models Trained

1. **Logistic Regression**: Interpretable baseline with WoE transformations
2. **XGBoost**: Gradient boosting for improved AUC (selected as production model)

### Model Selection

Best model selected by **ROC-AUC** on hold-out test set:
- XGBoost: **ROC-AUC 0.9996** (selected)
- Logistic Regression: ROC-AUC 0.9679

## 📈 Performance Metrics

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| **ROC-AUC** | **0.9996** | 0.9679 |
| **Precision** | **0.9957** | 0.7985 |
| **Recall** | **0.9708** | 0.9083 |
| **F1-Score** | **0.9831** | 0.8499 |

*Metrics tracked in MLflow - view detailed results with `mlflow ui`*

**Note**: These metrics are from the test set. The XGBoost model was selected as the production model based on superior ROC-AUC performance.

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) analysis provides:

- **Global Feature Importance**: Which features matter most overall
- **Local Explanations**: Why a specific customer received their risk score
- **Feature Interactions**: How features combine to affect predictions

View SHAP plots:
- In Streamlit dashboard: Navigate to "Model Explainability (SHAP)" page
- Generated plots: `artifacts/shap_plots/`

## 🚢 Deployment

### Docker

Build and run:
```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Docker Compose

```bash
docker compose up
```

### Environment Variables

- `MLFLOW_TRACKING_URI` (default: `file:./mlruns`)
- `MODEL_NAME` (default: `CreditRiskProxyModel`)
- `MODEL_STAGE` (default: `Production`)
- `PREDICTION_THRESHOLD` (default: `0.5`)
- `FEATURE_COLUMNS_PATH` (default: `/app/data/processed/expected_columns.json`)

## 🧪 Testing

Run test suite:
```bash
pytest -q
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## 📝 Development Guidelines

- Use feature branches and open PRs targeting `main`
- Add unit tests for new code; keep `pytest` passing
- Keep notebooks lightweight; move reusable logic into `src/`
- Do not commit data/models; prefer DVC or object storage
- Follow PEP 8 style guide (enforced via flake8 in CI)

## 🏛️ Basel II & Regulatory Compliance

**Basel II Alignment**: Capital requirements depend on accurate PD estimates. Our approach emphasizes:

- **Interpretability**: Transparent features and SHAP explanations
- **Documentation**: Full methodology and assumptions documented
- **Reproducibility**: Version-controlled code and MLflow experiment tracking
- **Governance**: Model registry, versioning, and monitoring hooks

**Proxy Target Risks**: Proxy-based labels require:
- Ongoing back-testing against real outcomes
- Stability monitoring across time windows
- Regular recalibration and refinement

## 📚 Key Concepts

### Proxy Target Necessity

With no direct default label, a proxy is required to train any supervised model. If the proxy poorly represents true default behavior, predictions can misprice risk, distort capital needs, and erode portfolio performance. Ongoing back-testing and proxy refinement are mandatory.

### Model Choice Trade-offs

- **Logistic Regression**: Transparent, stable, easy to govern but may sacrifice lift
- **Gradient Boosting (XGBoost)**: Improves AUC by modeling nonlinearity, yet raises validation burden, explainability costs (SHAP/PDP), and potential regulatory friction
- **Hybrid Approach**: Common pattern is an interpretable champion for regulatory use with a boosted challenger for monitoring

## 🔮 Future Enhancements

- [ ] Integrate real default outcomes for PD calibration
- [ ] Add survival analysis for time-to-default modeling
- [ ] Deploy monitoring dashboards (Prometheus + Grafana)
- [ ] Implement active learning for uncertain predictions
- [ ] Add sequence models for transaction history embeddings

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- **Author**: [Your Name]
- **10 Academy**: Week 12 Capstone Project

## 📧 Contact & Links

- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/YOUR_PROFILE)
- **Email**: your.email@example.com
- **Portfolio**: [Your Portfolio Website](https://your-portfolio.com)

---

## 📝 Setup Notes

**Before pushing to GitHub:**
1. Replace `YOUR_USERNAME` in the CI badge URL (line 3) with your actual GitHub username
2. Update clone URL (line 38) with your GitHub username
3. Add your contact information in the Contact section above
4. Deploy Streamlit dashboard and update Live Demo links (lines 68-69)
5. Generate screenshots and place in `artifacts/screenshots/` directory
