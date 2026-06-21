"""Generate the next 24h energy demand forecast using the MLflow champion model."""
import logging
import os

import holidays
import mlflow
import mlflow.prophet
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from psycopg2.extras import execute_values

from db.db_client import get_connection
from models.train_prophet import MODEL_NAME as PROPHET_MODEL_NAME
from models.train_xgboost import FEATURE_COLUMNS as XGB_FEATURE_COLUMNS
from models.train_xgboost import MODEL_NAME as XGB_MODEL_NAME

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HORIZON_HOURS = 24
HISTORY_LOOKBACK_HOURS = 48
LOCAL_TIMEZONE = "Europe/Zurich"
HOLIDAY_SUBDIVISION = "ZH"

# Try the XGBoost registry entry first, then fall back to Prophet.
_CANDIDATE_MODELS = (
    ("xgboost", XGB_MODEL_NAME, mlflow.xgboost.load_model),
    ("prophet", PROPHET_MODEL_NAME, mlflow.prophet.load_model),
)


def _latest_model_version(client: MlflowClient, name: str):
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def _load_champion_model():
    """Return (flavor, model, model_name, model_version) for the first
    registry entry that has a version and loads successfully."""
    client = MlflowClient()

    for flavor, name, loader in _CANDIDATE_MODELS:
        version = _latest_model_version(client, name)
        if version is None:
            logger.warning("No registered version found for model '%s'", name)
            continue
        try:
            model = loader(f"models:/{name}/{version.version}")
        except Exception:
            logger.exception("Failed to load model '%s' version %s", name, version.version)
            continue
        logger.info("Loaded champion model '%s' version %s (%s)", name, version.version, flavor)
        return flavor, model, name, version.version

    raise RuntimeError("No usable model found in the MLflow registry (tried xgboost, then prophet)")


def _read_recent_features(lookback_hours: int = HISTORY_LOOKBACK_HOURS) -> pd.DataFrame:
    """Read enough recent Gold features to seed lag/rolling features for forecasting."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, kwh, rolling_avg_7d
                FROM features
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (lookback_hours,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return pd.DataFrame(rows)


def _predict_xgboost(
    model, history: pd.DataFrame, horizon_hours: int = HORIZON_HOURS
) -> pd.DataFrame:
    """Recursively forecast `horizon_hours` ahead of the last known timestamp.

    lag_24h is always backed by real historical kwh (the 24h-ago timestamp for
    any point in the next 24h horizon falls within the already-known history).
    lag_1h chains onto the model's own prior-step prediction once we run past
    the last known actual hour. rolling_avg_7d carries forward the last known
    7-day rolling average, since 24 new hourly points barely move a 168h window.
    temperature is filled with 0, matching how missing weather is treated in
    training (no weather forecast source is wired up yet).
    """
    history = history.sort_values("timestamp").set_index("timestamp")
    kwh_lookup = history["kwh"].to_dict()
    last_rolling_avg_7d = history["rolling_avg_7d"].iloc[-1]
    last_ts = history.index.max()

    ch_holidays = holidays.CH(subdiv=HOLIDAY_SUBDIVISION, years=[last_ts.year, last_ts.year + 1])

    results = []
    for step in range(1, horizon_hours + 1):
        ts = last_ts + pd.Timedelta(hours=step)
        local_ts = ts.tz_convert(LOCAL_TIMEZONE)
        feature_row = {
            "hour": local_ts.hour,
            "day_of_week": local_ts.dayofweek,
            "month": local_ts.month,
            "is_weekend": int(local_ts.dayofweek in (5, 6)),
            "is_holiday": int(local_ts.date() in ch_holidays),
            "lag_1h": kwh_lookup[ts - pd.Timedelta(hours=1)],
            "lag_24h": kwh_lookup[ts - pd.Timedelta(hours=24)],
            "rolling_avg_7d": last_rolling_avg_7d,
            "temperature": 0.0,
        }
        features = pd.DataFrame([feature_row])[XGB_FEATURE_COLUMNS]
        y_pred = float(model.predict(features)[0])
        kwh_lookup[ts] = y_pred  # feed forward so the next step's lag_1h sees this prediction
        results.append({"timestamp": ts, "predicted_kwh": y_pred})

    return pd.DataFrame(results)


def _predict_prophet(
    model, last_ts: pd.Timestamp, horizon_hours: int = HORIZON_HOURS
) -> pd.DataFrame:
    future_ts = [last_ts + pd.Timedelta(hours=i) for i in range(1, horizon_hours + 1)]
    future = pd.DataFrame({"ds": [t.tz_localize(None) for t in future_ts]})
    forecast = model.predict(future)
    return pd.DataFrame({"timestamp": future_ts, "predicted_kwh": forecast["yhat"].to_numpy()})


def _insert_predictions(conn, df: pd.DataFrame, model_name: str, model_version: str) -> None:
    rows = [
        (ts, model_name, model_version, float(kwh))
        for ts, kwh in zip(df["timestamp"], df["predicted_kwh"])
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO predictions (timestamp, model_name, model_version, predicted_kwh)
            VALUES %s
            """,
            rows,
        )
    conn.commit()


def predict() -> list:
    """Forecast the next 24 hours with the MLflow champion model and persist the result."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    try:
        flavor, model, model_name, model_version = _load_champion_model()
    except RuntimeError:
        logger.exception("predict: no champion model available")
        return []

    history = _read_recent_features()
    if history.empty:
        logger.error("predict: features table is empty; cannot seed a forecast")
        return []

    if flavor == "xgboost":
        forecast_df = _predict_xgboost(model, history)
    else:
        forecast_df = _predict_prophet(model, history["timestamp"].max())

    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        _insert_predictions(conn, forecast_df, model_name, model_version)
    finally:
        conn.close()

    logger.info(
        "predict: inserted %d prediction rows from '%s' v%s",
        len(forecast_df), model_name, model_version,
    )

    return forecast_df.to_dict("records")


if __name__ == "__main__":
    predict()
