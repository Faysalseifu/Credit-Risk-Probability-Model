import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Configuration
_default_uri = "file:./notebooks/mlruns" if Path("notebooks/mlruns").exists() else "file:./mlruns"
mlflow.set_tracking_uri(_default_uri)
client = MlflowClient()

model_name = "CreditRiskProxyModel"
version = 1

print(f"Transitioning {model_name} version {version} to Production...")
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production"
)
print("✅ Done!")
