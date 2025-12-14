# Credit Risk Probability Model (Alternative Data)
End-to-end pipeline for building, training, and deploying a credit-risk scoring service for Bati Bank.

## Project structure
```
credit-risk-model/
├── data/                        # Git-ignored raw & processed data
│   ├── raw/                     # Raw data files (e.g., data.csv)
│   └── processed/               # Cleaned / feature-engineered data
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering utilities
│   ├── train.py                 # Model training script (CLI)
│   ├── predict.py               # Batch inference script (CLI)
│   └── api/
│       ├── main.py              # FastAPI app exposing prediction endpoint
│       └── pydantic_models.py   # Request/response schemas
├── tests/
│   └── test_data_processing.py  # Unit tests
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Local orchestration (API + model)
├── requirements.txt             # Python dependencies
└── README.md                    # Project docs
```

## Quickstart
1) Install dependencies: `pip install -r requirements.txt`
2) Run tests: `pytest -q`
3) Explore data: open notebooks/eda.ipynb and run all cells.
4) Build image: `docker compose build`
5) Run API locally (hot reload): `docker compose up`

## Model training and tracking (Task 5)
- Optional: launch MLflow UI with `mlflow ui` (default http://127.0.0.1:5000).
- Train and log models (LogReg, Random Forest, GBM):
```
python -m src.train \
    --raw-path data/raw/data.csv \
    --model-out artifacts/best_model.pkl
```
- The grid search selects the best hyper-parameters (AUC on a hold-out set) and registers the top model under the MLflow Model Registry name `credit-risk-best`.

## Development guidelines
- Use feature branches and open PRs targeting main.
- Add unit tests for new code; keep `pytest` passing.
- Keep notebooks lightweight; move reusable logic into src/.
- Do not commit data/models; prefer DVC or object storage.

## Credit Scoring Business Understanding
**Basel II and interpretability**: Capital depends on PD estimates, so regulators must retrace every assumption. We use interpretable features, monotonic scorecard-style transformations, and full documentation so audits can reproduce results and challenge drivers.

**Proxy target necessity and risks**: With no direct default label, a proxy is required to train any supervised model. If the proxy poorly represents true default behavior, predictions can misprice risk, distort capital needs, and erode portfolio performance; ongoing back-testing and proxy refinement are mandatory.

**Model choice trade-offs**: Logistic Regression with WoE is transparent, stable, and easy to govern but may sacrifice lift. Gradient Boosting improves AUC by modeling nonlinearity, yet raises validation burden, explainability costs (SHAP/PDP), and potential regulatory friction. A common pattern is an interpretable champion for regulatory use with a boosted challenger for monitoring.