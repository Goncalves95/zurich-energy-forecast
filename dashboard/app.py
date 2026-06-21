"""Streamlit dashboard: actual vs forecast energy demand."""
import streamlit as st

st.set_page_config(page_title="Zürich Energy Forecast", page_icon="⚡", layout="wide")

st.title("⚡ Zürich Energy Forecast")
st.caption("Real-time energy demand forecasting — next 24 hours")

# TODO: load data from DB and render charts
st.info("Pipeline not yet connected. Run the ingestion and training steps first.")
