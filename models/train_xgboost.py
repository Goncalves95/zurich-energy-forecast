"""Train XGBoost regressor for 24h energy demand forecasting."""
import logging
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from db.db_client import get_connection  # noqa: E402
from models.evaluate import compute_metrics  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "xgboost-energy-zurich"
EXPERIMENT_NAME = "zurich-energy-forecast"
TEST_WINDOW_DAYS = 30

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1h",
    "lag_24h",
    "rolling_avg_7d",
    "temperature",
]
TARGET_COLUMN = "kwh"

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
}


def _read_features() -> pd.DataFrame:
    query = (
        f"SELECT timestamp, {TARGET_COLUMN}, {', '.join(FEATURE_COLUMNS)} "
        "FROM features ORDER BY timestamp"
    )
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame(rows)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose lag/rolling features aren't available yet, and fill
    missing temperature (sparse weather coverage) with 0."""
    df = df.dropna(subset=["lag_1h", "lag_24h", "rolling_avg_7d"]).copy()
    df["temperature"] = df["temperature"].fillna(0)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    return df


def split_train_test(df: pd.DataFrame, test_window_days: int = TEST_WINDOW_DAYS):
    """Last `test_window_days` days (by timestamp) become the test set."""
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_window_days)
    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]
    return train, test


def _log_feature_importance_plot(model: XGBRegressor, feature_names: list) -> None:
    importances = model.feature_importances_
    order = importances.argsort()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in order], importances[order])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "feature_importance.png")
        fig.savefig(path)
        mlflow.log_artifact(path)
    plt.close(fig)


def train() -> dict:
    """Train an XGBoost regressor on Gold features and register it in MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = _read_features()
    if df.empty:
        raise RuntimeError("features table is empty; run the pipeline before training")

    df = _prepare(df)
    train_df, test_df = split_train_test(df)
    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"Not enough data for a {TEST_WINDOW_DAYS}-day test split "
            f"({len(train_df)} train rows, {len(test_df)} test rows)"
        )

    x_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    x_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = XGBRegressor(**XGB_PARAMS, random_state=42)

    with mlflow.start_run(run_name="xgboost-energy-zurich") as run:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("test_window_days", TEST_WINDOW_DAYS)
        mlflow.log_metrics(metrics)

        _log_feature_importance_plot(model, FEATURE_COLUMNS)

        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        logger.info("xgboost training run %s: %s", run.info.run_id, metrics)

    return metrics


if __name__ == "__main__":
    train()
