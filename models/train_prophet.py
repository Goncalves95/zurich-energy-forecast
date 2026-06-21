"""Train Prophet model with Swiss holiday calendar."""
import logging
import os

import holidays
import mlflow
import mlflow.prophet
import pandas as pd
from dotenv import load_dotenv
from prophet import Prophet

from db.db_client import get_connection
from models.evaluate import compute_metrics

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "prophet-energy-zurich"
EXPERIMENT_NAME = "zurich-energy-forecast"
TEST_WINDOW_DAYS = 30
HOLIDAY_SUBDIVISION = "ZH"

PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": True,
}


def _read_features() -> pd.DataFrame:
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT timestamp, kwh FROM features ORDER BY timestamp")
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.rename(columns={"timestamp": "ds", "kwh": "y"})


def _ch_holidays_df(years: list) -> pd.DataFrame:
    ch_holidays = holidays.CH(subdiv=HOLIDAY_SUBDIVISION, years=years)
    return pd.DataFrame({
        "holiday": "ch_zh_holiday",
        "ds": pd.to_datetime(list(ch_holidays.keys())),
    })


def split_train_test(df: pd.DataFrame, test_window_days: int = TEST_WINDOW_DAYS):
    """Last `test_window_days` days (by ds) become the test set."""
    cutoff = df["ds"].max() - pd.Timedelta(days=test_window_days)
    train = df[df["ds"] <= cutoff]
    test = df[df["ds"] > cutoff]
    return train, test


def train() -> dict:
    """Train a Prophet model on Gold features and register it in MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = _read_features()
    if df.empty:
        raise RuntimeError("features table is empty; run the pipeline before training")

    # Prophet requires tz-naive timestamps.
    df = df.assign(ds=df["ds"].dt.tz_localize(None))

    train_df, test_df = split_train_test(df)
    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"Not enough data for a {TEST_WINDOW_DAYS}-day test split "
            f"({len(train_df)} train rows, {len(test_df)} test rows)"
        )

    years = sorted(set(train_df["ds"].dt.year) | set(test_df["ds"].dt.year))
    holidays_df = _ch_holidays_df(years)

    model = Prophet(holidays=holidays_df, **PROPHET_PARAMS)

    with mlflow.start_run(run_name="prophet-energy-zurich") as run:
        model.fit(train_df[["ds", "y"]])

        forecast = model.predict(test_df[["ds"]])
        metrics = compute_metrics(test_df["y"].to_numpy(), forecast["yhat"].to_numpy())

        mlflow.log_params(PROPHET_PARAMS)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("test_window_days", TEST_WINDOW_DAYS)
        mlflow.log_metrics(metrics)

        mlflow.prophet.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        logger.info("prophet training run %s: %s", run.info.run_id, metrics)

    return metrics


if __name__ == "__main__":
    train()
