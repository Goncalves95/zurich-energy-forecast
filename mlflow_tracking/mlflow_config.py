"""MLflow experiment setup and model registry helpers."""
import mlflow

EXPERIMENT_NAME = "zurich-energy-forecast"
TRACKING_URI = "http://localhost:5000"


def get_experiment() -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return EXPERIMENT_NAME
