import mlflow.pyfunc
from pathlib import Path
import os

_default_uri = "file:./notebooks/mlruns" if Path("notebooks/mlruns").exists() else "file:./mlruns"
mlflow.set_tracking_uri(_default_uri)
model_uri = "models:/CreditRiskProxyModel/Production"
model = mlflow.pyfunc.load_model(model_uri)

print(f"Model type: {type(model)}")
print(f"Model implementation type: {type(model._model_impl)}")

# Try to find the booster
if hasattr(model._model_impl, 'python_model'):
    print("Found python_model")
    pm = model._model_impl.python_model
    print(f"Python model type: {type(pm)}")
    if hasattr(pm, 'model'):
        print(f"Internal model type: {type(pm.model)}")
else:
    print("No python_model attribute")
    # List all attributes
    print(f"Attributes: {dir(model._model_impl)}")
    if hasattr(model._model_impl, '_model'):
        print(f"Found _model type: {type(model._model_impl._model)}")
