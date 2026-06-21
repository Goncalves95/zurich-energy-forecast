"""Streamlit dashboard: actual vs forecast energy demand."""
import logging
import os
import time

import mlflow
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from mlflow import MlflowClient

from db.db_client import get_connection
from models.train_prophet import MODEL_NAME as PROPHET_MODEL_NAME
from models.train_xgboost import MODEL_NAME as XGBOOST_MODEL_NAME

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Zürich Energy Forecast", page_icon="⚡", layout="wide")

LOCAL_TIMEZONE = "Europe/Zurich"
CHART_LOOKBACK_DAYS = 7
CACHE_TTL_SECONDS = 300
REFRESH_INTERVAL_SECONDS = 300
ALL_TABLES = ["raw_energy", "clean_energy", "features", "predictions"]
# Tried in champion order: XGBoost first, Prophet as fallback (matches models/predict.py).
CANDIDATE_MODEL_NAMES = (XGBOOST_MODEL_NAME, PROPHET_MODEL_NAME)


# ---------------------------------------------------------------------------
# Data access (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_table_counts() -> dict:
    """Row counts for each pipeline table; None for a table if the query fails."""
    counts = {table: None for table in ALL_TABLES}
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        return counts

    try:
        with conn.cursor() as cur:
            for table in ALL_TABLES:
                cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
                counts[table] = cur.fetchone()["n"]
    except psycopg2.Error:
        logger.exception("Failed to read table counts")
    finally:
        conn.close()
    return counts


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_latest_timestamp(table: str):
    """Latest `timestamp` value in the given table, or None if empty/unreachable."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT MAX(timestamp) AS ts FROM {table}")
            row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to read latest timestamp from %s", table)
        return None
    finally:
        conn.close()
    return row["ts"] if row else None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_latest_actual() -> dict | None:
    """Most recent clean_energy reading: {timestamp, kwh}."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT timestamp, kwh FROM clean_energy ORDER BY timestamp DESC LIMIT 1")
            row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to read latest clean_energy row")
        return None
    finally:
        conn.close()
    return dict(row) if row else None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_next_hour_forecast() -> dict | None:
    """Soonest still-upcoming row from the most recently generated prediction batch."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, predicted_kwh
                FROM predictions
                WHERE created_at = (SELECT MAX(created_at) FROM predictions)
                  AND timestamp > NOW()
                ORDER BY timestamp ASC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to read next-hour forecast")
        return None
    finally:
        conn.close()
    return dict(row) if row else None


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_actual_last_n_days(days: int = CHART_LOOKBACK_DAYS) -> pd.DataFrame:
    """clean_energy readings from the last `days` days."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        return pd.DataFrame()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, kwh
                FROM clean_energy
                WHERE timestamp >= NOW() - INTERVAL '1 day' * %s
                ORDER BY timestamp
                """,
                (days,),
            )
            rows = cur.fetchall()
    except psycopg2.Error:
        logger.exception("Failed to read clean_energy history")
        return pd.DataFrame()
    finally:
        conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_latest_prediction_batch() -> pd.DataFrame:
    """All rows from the most recently generated prediction batch (one predict()
    run inserts a same-created_at batch, typically the next 24 hours)."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        return pd.DataFrame()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, predicted_kwh, model_name, model_version
                FROM predictions
                WHERE created_at = (SELECT MAX(created_at) FROM predictions)
                ORDER BY timestamp
                """
            )
            rows = cur.fetchall()
    except psycopg2.Error:
        logger.exception("Failed to read predictions")
        return pd.DataFrame()
    finally:
        conn.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_champion_model_info() -> dict | None:
    """{name, version, mape} for the first candidate model with a registered
    version (xgboost first, prophet fallback — matches models/predict.py)."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    try:
        client = MlflowClient()
    except Exception:
        logger.exception("Could not create an MLflow client")
        return None

    for name in CANDIDATE_MODEL_NAMES:
        try:
            versions = client.search_model_versions(f"name='{name}'")
        except Exception:
            logger.exception("Failed to query MLflow registry for model '%s'", name)
            continue
        if not versions:
            continue

        latest = max(versions, key=lambda v: int(v.version))
        mape = None
        if latest.run_id:
            try:
                run = client.get_run(latest.run_id)
                mape = run.data.metrics.get("mape")
            except Exception:
                logger.exception("Failed to fetch MLflow run %s", latest.run_id)
        return {"name": name, "version": latest.version, "mape": mape}

    return None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.title("⚡ Zürich Energy Forecast")
    now_local = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
    st.caption(f"As of {now_local.strftime('%A, %d %B %Y %H:%M:%S %Z')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        latest = get_latest_actual()
        if latest:
            ts_local = pd.Timestamp(latest["timestamp"]).tz_convert(LOCAL_TIMEZONE)
            st.metric("Latest Actual Consumption", f"{latest['kwh']:,.0f} kWh")
            st.caption(f"as of {ts_local.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.metric("Latest Actual Consumption", "N/A")
            st.caption("No data in clean_energy yet")

    with col2:
        forecast = get_next_hour_forecast()
        if forecast:
            ts_local = pd.Timestamp(forecast["timestamp"]).tz_convert(LOCAL_TIMEZONE)
            st.metric("Next Hour Forecast", f"{forecast['predicted_kwh']:,.0f} kWh")
            st.caption(f"for {ts_local.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.metric("Next Hour Forecast", "N/A")
            st.caption("No upcoming predictions yet")

    with col3:
        model_info = get_champion_model_info()
        if model_info and model_info["mape"] is not None:
            st.metric("Model Accuracy (MAPE)", f"{model_info['mape']:.2f}%")
            st.caption(f"{model_info['name']} v{model_info['version']}")
        else:
            st.metric("Model Accuracy (MAPE)", "N/A")
            st.caption("No MLflow run metrics available")


def render_main_chart() -> None:
    st.subheader("Actual vs Forecast (Last 7 Days)")

    actual_df = get_actual_last_n_days(CHART_LOOKBACK_DAYS)
    predicted_df = get_latest_prediction_batch()

    if actual_df.empty and predicted_df.empty:
        st.info("No actual or forecast data yet — run the pipeline and prediction steps first.")
        return

    fig = go.Figure()

    if not actual_df.empty:
        actual_local_ts = actual_df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        fig.add_trace(go.Scatter(
            x=actual_local_ts,
            y=actual_df["kwh"],
            mode="lines",
            name="Actual",
            line={"color": "blue"},
        ))
    else:
        st.info("No actual consumption data in the last 7 days yet.")

    if not predicted_df.empty:
        predicted_local_ts = predicted_df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        fig.add_trace(go.Scatter(
            x=predicted_local_ts,
            y=predicted_df["predicted_kwh"],
            mode="lines",
            name="Forecast",
            line={"color": "orange", "dash": "dash"},
        ))
    else:
        st.info("No forecast data yet — run `python -m models.predict` to generate one.")

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="kWh",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 30},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sidebar() -> None:
    st.sidebar.header("Pipeline Status")

    last_ingestion = get_latest_timestamp("raw_energy")
    last_pipeline_run = get_latest_timestamp("features")

    st.sidebar.markdown("**Last ingestion run**")
    if last_ingestion is not None:
        ts_local = pd.Timestamp(last_ingestion).tz_convert(LOCAL_TIMEZONE)
        st.sidebar.write(ts_local.strftime("%Y-%m-%d %H:%M"))
    else:
        st.sidebar.write("No data yet")

    st.sidebar.markdown("**Last pipeline run**")
    if last_pipeline_run is not None:
        st.sidebar.write(
            pd.Timestamp(last_pipeline_run).tz_convert(LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M")
        )
    else:
        st.sidebar.write("No data yet")

    st.sidebar.markdown("**Total records**")
    counts = get_table_counts()
    for table in ALL_TABLES:
        count = counts.get(table)
        st.sidebar.write(f"- {table}: {count:,}" if count is not None else f"- {table}: N/A")

    st.sidebar.markdown("**Model info**")
    model_info = get_champion_model_info()
    if model_info:
        st.sidebar.write(f"{model_info['name']} (v{model_info['version']})")
    else:
        st.sidebar.write("No registered model yet")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_header()
st.divider()
render_main_chart()
render_sidebar()

refresh_placeholder = st.empty()
with refresh_placeholder.container():
    st.caption(f"Auto-refreshing every {REFRESH_INTERVAL_SECONDS // 60} minutes.")
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()
