#!/bin/bash
# zurich-energy-forecast — Project Scaffold
# Run this from the root of your cloned repository:
#   bash scaffold.sh

set -e

echo "🏗️  Scaffolding zurich-energy-forecast..."

# ── Directories ──────────────────────────────────────────────────────────────
mkdir -p data/raw
mkdir -p data/samples

mkdir -p ingestion
mkdir -p pipeline
mkdir -p models
mkdir -p mlflow_tracking
mkdir -p dashboard
mkdir -p db
mkdir -p monitoring
mkdir -p tests
mkdir -p .github/workflows

# ── Ingestion ─────────────────────────────────────────────────────────────────
cat > ingestion/__init__.py << 'EOF'
EOF

cat > ingestion/fetch_energy.py << 'EOF'
"""Fetch hourly electricity consumption data from Open Data Zürich (ewz)."""


def fetch_energy() -> None:
    # TODO: implement API call to data.stadt-zuerich.ch
    pass


if __name__ == "__main__":
    fetch_energy()
EOF

cat > ingestion/fetch_weather.py << 'EOF'
"""Fetch hourly weather data from MeteoSwiss."""


def fetch_weather() -> None:
    # TODO: implement API call to MeteoSwiss
    pass


if __name__ == "__main__":
    fetch_weather()
EOF

cat > ingestion/scheduler.py << 'EOF'
"""Cron-based scheduler for ingestion jobs (Prefect-ready)."""
EOF

# ── Pipeline ──────────────────────────────────────────────────────────────────
cat > pipeline/__init__.py << 'EOF'
EOF

cat > pipeline/bronze_to_silver.py << 'EOF'
"""Transform raw (Bronze) data to cleaned (Silver) layer."""


def run() -> None:
    # TODO: null handling, deduplication, range validation
    pass


if __name__ == "__main__":
    run()
EOF

cat > pipeline/silver_to_gold.py << 'EOF'
"""Transform clean (Silver) data to ML-ready features (Gold layer)."""


def run() -> None:
    # TODO: lag features, rolling averages, cyclic time encoding
    pass


if __name__ == "__main__":
    run()
EOF

cat > pipeline/validate.py << 'EOF'
"""Data quality checks at each layer transition."""


def validate_bronze() -> bool:
    # TODO: timestamp continuity, null rate checks
    return True


def validate_silver() -> bool:
    # TODO: range checks, type enforcement
    return True
EOF

# ── Models ────────────────────────────────────────────────────────────────────
cat > models/__init__.py << 'EOF'
EOF

cat > models/train_xgboost.py << 'EOF'
"""Train XGBoost regressor for 24h energy demand forecasting."""


def train() -> None:
    # TODO: load Gold features, train, log to MLflow
    pass


if __name__ == "__main__":
    train()
EOF

cat > models/train_prophet.py << 'EOF'
"""Train Prophet model with Swiss holiday calendar."""


def train() -> None:
    # TODO: load Gold features, add CH/ZH holidays, train, log to MLflow
    pass


if __name__ == "__main__":
    train()
EOF

cat > models/evaluate.py << 'EOF'
"""Evaluate model performance: RMSE, MAE, R² and business metric translation."""


def evaluate(model_name: str) -> dict:
    # TODO: load model from MLflow registry, run evaluation
    return {}
EOF

cat > models/predict.py << 'EOF'
"""Generate next 24h energy demand forecast."""


def predict() -> list:
    # TODO: load champion model, run inference
    return []


if __name__ == "__main__":
    predict()
EOF

# ── MLflow ────────────────────────────────────────────────────────────────────
cat > mlflow_tracking/__init__.py << 'EOF'
EOF

cat > mlflow_tracking/mlflow_config.py << 'EOF'
"""MLflow experiment setup and model registry helpers."""
import mlflow

EXPERIMENT_NAME = "zurich-energy-forecast"
TRACKING_URI = "http://localhost:5000"


def get_experiment() -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return EXPERIMENT_NAME
EOF

# ── Dashboard ─────────────────────────────────────────────────────────────────
cat > dashboard/__init__.py << 'EOF'
EOF

cat > dashboard/app.py << 'EOF'
"""Streamlit dashboard: actual vs forecast energy demand."""
import streamlit as st

st.set_page_config(page_title="Zürich Energy Forecast", page_icon="⚡", layout="wide")

st.title("⚡ Zürich Energy Forecast")
st.caption("Real-time energy demand forecasting — next 24 hours")

# TODO: load data from DB and render charts
st.info("Pipeline not yet connected. Run the ingestion and training steps first.")
EOF

# ── Database ──────────────────────────────────────────────────────────────────
cat > db/__init__.py << 'EOF'
EOF

cat > db/schema.sql << 'EOF'
-- ============================================================
-- Zürich Energy Forecast — PostgreSQL Schema
-- Bronze / Silver / Gold layers
-- ============================================================

-- BRONZE: raw ingested data
CREATE TABLE IF NOT EXISTS raw_energy (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    kwh         DOUBLE PRECISION,
    source      TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_weather (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMPTZ NOT NULL,
    temperature  DOUBLE PRECISION,
    humidity     DOUBLE PRECISION,
    solar_rad    DOUBLE PRECISION,
    source       TEXT,
    ingested_at  TIMESTAMPTZ DEFAULT NOW()
);

-- SILVER: cleaned data
CREATE TABLE IF NOT EXISTS clean_energy (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL UNIQUE,
    kwh         DOUBLE PRECISION NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clean_weather (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMPTZ NOT NULL UNIQUE,
    temperature  DOUBLE PRECISION NOT NULL,
    humidity     DOUBLE PRECISION,
    solar_rad    DOUBLE PRECISION,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- GOLD: ML-ready feature store
CREATE TABLE IF NOT EXISTS features (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL UNIQUE,
    kwh             DOUBLE PRECISION NOT NULL,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    solar_rad       DOUBLE PRECISION,
    hour            INTEGER,
    day_of_week     INTEGER,
    month           INTEGER,
    is_weekend      BOOLEAN,
    is_holiday      BOOLEAN,
    lag_1h          DOUBLE PRECISION,
    lag_24h         DOUBLE PRECISION,
    rolling_avg_7d  DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- PREDICTIONS
CREATE TABLE IF NOT EXISTS predictions (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMPTZ NOT NULL,
    model_name   TEXT NOT NULL,
    model_version TEXT,
    predicted_kwh DOUBLE PRECISION NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
EOF

cat > db/db_client.py << 'EOF'
"""PostgreSQL connection and query helpers."""
import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "energy_forecast"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        cursor_factory=RealDictCursor,
    )
EOF

# ── Monitoring ────────────────────────────────────────────────────────────────
cat > monitoring/__init__.py << 'EOF'
EOF

cat > monitoring/drift_detector.py << 'EOF'
"""Detect data drift in feature distributions."""


def detect_drift(reference_period: str, current_period: str) -> dict:
    # TODO: compare feature distributions using KS test or PSI
    return {"drift_detected": False}
EOF

# ── Tests ─────────────────────────────────────────────────────────────────────
cat > tests/__init__.py << 'EOF'
EOF

cat > tests/test_ingestion.py << 'EOF'
"""Tests for ingestion layer."""


def test_fetch_energy_returns_none():
    from ingestion.fetch_energy import fetch_energy
    assert fetch_energy() is None


def test_fetch_weather_returns_none():
    from ingestion.fetch_weather import fetch_weather
    assert fetch_weather() is None
EOF

cat > tests/test_pipeline.py << 'EOF'
"""Tests for pipeline transformations."""


def test_bronze_to_silver_runs():
    from pipeline.bronze_to_silver import run
    assert run() is None


def test_validate_bronze_passes():
    from pipeline.validate import validate_bronze
    assert validate_bronze() is True
EOF

cat > tests/test_models.py << 'EOF'
"""Tests for model training and prediction."""


def test_predict_returns_list():
    from models.predict import predict
    result = predict()
    assert isinstance(result, list)
EOF

# ── GitHub Actions ────────────────────────────────────────────────────────────
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint (Ruff)
        run: ruff check .

      - name: Type check (MyPy)
        run: mypy . --ignore-missing-imports

      - name: Run tests
        run: pytest tests/ -v --cov=.
EOF

cat > .github/workflows/retrain.yml << 'EOF'
name: Weekly Model Retrain

on:
  schedule:
    - cron: "0 3 * * 1"   # Every Monday at 03:00 UTC
  workflow_dispatch:         # Allow manual trigger

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run XGBoost training
        run: python models/train_xgboost.py

      - name: Run Prophet training
        run: python models/train_prophet.py

      - name: Evaluate models
        run: python models/evaluate.py
EOF

# ── Docker ────────────────────────────────────────────────────────────────────
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ingestion/scheduler.py"]
EOF

cat > docker-compose.yml << 'EOF'
version: "3.9"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: energy_forecast
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/schema.sql:/docker-entrypoint-initdb.d/schema.sql

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.12.1
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - mlflow_data:/mlflow

  dashboard:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    environment:
      DB_HOST: postgres
      DB_PORT: "5432"
      DB_NAME: energy_forecast
      DB_USER: postgres
      DB_PASSWORD: postgres
    depends_on:
      - postgres

volumes:
  postgres_data:
  mlflow_data:
EOF

# ── Config files ──────────────────────────────────────────────────────────────
cat > requirements.txt << 'EOF'
# Data
pandas==2.2.2
numpy==1.26.4
requests==2.32.3

# ML
scikit-learn==1.5.0
xgboost==2.0.3
prophet==1.1.5

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.30

# MLflow
mlflow==2.12.1

# Dashboard
streamlit==1.35.0
plotly==5.22.0

# Orchestration
prefect==2.19.7

# Quality
ruff==0.4.4
mypy==1.10.0
pytest==8.2.1
pytest-cov==5.0.0
EOF

cat > pyproject.toml << 'EOF'
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
EOF

cat > .env.example << 'EOF'
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=energy_forecast
DB_USER=postgres
DB_PASSWORD=postgres

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# APIs
OPEN_DATA_ZURICH_TOKEN=your_token_here
METEOSWISS_API_KEY=your_key_here
EOF

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
dist/
build/

# Environment
.env

# Data (never commit raw data)
data/raw/
data/samples/

# MLflow artifacts
mlruns/
mlflow_artifacts/

# IDE
.vscode/
.idea/
*.DS_Store

# Docker
*.log
EOF

echo ""
echo "✅ Scaffold complete! Structure created:"
echo ""
find . -not -path './.git/*' -not -name '.git' | sort | sed 's|[^/]*/|  |g'
echo ""
echo "👉 Next steps:"
echo "   1. cp .env.example .env  (and fill in your values)"
echo "   2. docker-compose up -d  (start PostgreSQL + MLflow + Dashboard)"
echo "   3. git add . && git commit -m 'chore: initial project scaffold'"