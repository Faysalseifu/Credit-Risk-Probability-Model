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

## Credit scoring business context
**Basel II and interpretability**: Under the IRB approach, supervisors must trace every input to the PD estimate and its impact on capital. We prioritize interpretable features, monotonic relationships, and reproducible documentation.

**Proxy target and its risks**: The dataset lacks an explicit default label, so we build a proxy (e.g., high-risk behavior). A weak proxy can misstate risk and pricing; continuous back-testing against true defaults is required once available.

**Simple vs. complex models**: Logistic regression/scorecards are transparent and easy to validate; ensembles (GBM, RF) capture nonlinearity but need post-hoc explainability (SHAP, PDP) and add validation burden. A common pattern is a simple champion for regulatory use with a complex challenger for monitoring.