# ⚡ Zürich Energy Forecast

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **End-to-end MLOps pipeline for real-time energy demand forecasting in Zürich.**  
> Ingests public data from Open Data Zürich and MeteoSwiss, processes it through a layered data architecture, and forecasts energy demand for the next 24 hours using XGBoost and Prophet — with full model tracking, automated retraining, and a live dashboard.

---

## 📌 Motivation

Zürich is one of Europe's leading smart cities, committed to ambitious energy transition targets. Accurate short-term energy demand forecasting enables:

- **Grid optimisation** — prevent overload during peak hours
- **Cost reduction** — purchase energy at lower prices in off-peak windows
- **Sustainability** — reduce reliance on carbon-intensive backup generation

This project demonstrates how real public data, robust data engineering, and production-grade ML practices can be combined to deliver measurable value for a city like Zürich.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                               │
│   Open Data Zürich (ewz)   ·   MeteoSwiss API   ·   Public Holidays│
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Hourly ingestion (cron / Prefect)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BRONZE LAYER (Raw)                           │
│            PostgreSQL — raw_energy, raw_weather tables              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Cleaning · Validation · Deduplication
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SILVER LAYER (Clean)                         │
│         PostgreSQL — clean_energy, clean_weather tables             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Feature engineering
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GOLD LAYER (Feature Store)                      │
│              PostgreSQL — features table (ML-ready)                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴───────────────┐
              ▼                                ▼
     ┌────────────────┐              ┌──────────────────┐
     │  XGBoost Model │              │  Prophet Model   │
     │  (tabular ML)  │              │  (seasonality)   │
     └────────┬───────┘              └────────┬─────────┘
              └────────────┬──────────────────┘
                           ▼
              ┌────────────────────────┐
              │   MLflow Model Registry│
              │  Experiment Tracking   │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  Streamlit Dashboard   │
              │  Actual vs Forecast    │
              │  Next 24h prediction   │
              └────────────────────────┘
```

---

## 🗂️ Project Structure

```
zurich-energy-forecast/
│
├── data/                        # Local data samples (not committed — see .gitignore)
│
├── ingestion/                   # Data ingestion layer
│   ├── fetch_energy.py          # Pulls ewz consumption data from Open Data Zürich
│   ├── fetch_weather.py         # Pulls temperature/humidity from MeteoSwiss
│   └── scheduler.py             # Cron-based orchestration (Prefect-ready)
│
├── pipeline/                    # Data transformation pipeline
│   ├── bronze_to_silver.py      # Cleaning, null handling, deduplication
│   ├── silver_to_gold.py        # Feature engineering (lag, rolling avg, cyclic encoding)
│   └── validate.py              # Data quality checks (Great Expectations)
│
├── models/                      # ML training and evaluation
│   ├── train_xgboost.py         # XGBoost regressor with feature importance
│   ├── train_prophet.py         # Prophet model with Swiss holiday calendar
│   ├── evaluate.py              # RMSE, MAE, R² + business metric translation
│   └── predict.py               # Inference script for next 24h
│
├── mlflow/                      # MLflow configuration
│   └── mlflow_config.py         # Experiment setup and model registry
│
├── dashboard/                   # Streamlit visualisation app
│   └── app.py                   # Live dashboard: actual vs forecast
│
├── db/                          # Database layer
│   ├── schema.sql               # Schema reference (Bronze/Silver/Gold) — not applied directly
│   ├── db_client.py             # DB connection and query helpers
│   └── migrations/              # Alembic migrations (`alembic upgrade head` to apply)
│       └── versions/
│
├── alembic.ini                   # Alembic config (reads DATABASE_URL from .env)
│
├── monitoring/                  # Model monitoring
│   └── drift_detector.py        # Data drift detection (feature distribution shift)
│
├── tests/                       # Unit and integration tests
├── monitoring/                  # Model monitoring
│   └── drift_detector.py        # Data drift detection (feature distribution shift)
│
├── tests/                       # Unit and integration tests
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_models.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Lint, test, type-check on every PR
│       └── retrain.yml          # Scheduled weekly model retraining
│
├── docker-compose.yml           # PostgreSQL + MLflow + App services
├── Dockerfile                   # Production container for ingestion + model
├── requirements.txt
├── pyproject.toml               # Ruff + MyPy config
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | XGBoost · Prophet · Scikit-Learn |
| Data Storage | PostgreSQL 16 (Bronze / Silver / Gold layers) |
| Experiment Tracking | MLflow (local or remote) |
| Orchestration | Prefect (or Cron + Docker) |
| Containerisation | Docker · Docker Compose |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit |
| Code Quality | Ruff · MyPy · Pytest |

---

## 📊 Data Sources

| Source | Data | Update Frequency |
|---|---|---|
| [Open Data Zürich](https://data.stadt-zuerich.ch/) | ewz electricity consumption | Hourly |
| [MeteoSwiss](https://www.meteoswiss.admin.ch/) | Temperature, humidity, solar radiation | Hourly |
| [Swiss Federal Holidays](https://www.feiertagskalender.ch/) | Public holidays (CH/ZH) | Annual |

---

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- PostgreSQL 16 (or use the provided Docker service)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/zurich-energy-forecast.git
cd zurich-energy-forecast
```

### 2. Start services

```bash
docker-compose up -d
```

This starts PostgreSQL, MLflow tracking server, and the Streamlit dashboard.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialise the database

Schema changes are managed with [Alembic](https://alembic.sqlalchemy.org/) migrations under `db/migrations/`, not by re-running `db/schema.sql` or `db/init_db.py` directly.

```bash
alembic upgrade head
```

This reads `DATABASE_URL` from `.env` (see `db/migrations/env.py`). `db/schema.sql` is kept only as a human-readable reference of the current schema; to change the schema, add a new migration with `alembic revision -m "<description>"` instead of editing `schema.sql` and re-running `init_db.py`.

### 5. Run the ingestion pipeline

```bash
python ingestion/fetch_energy.py
python ingestion/fetch_weather.py
```

### 6. Run the full pipeline

```bash
python pipeline/bronze_to_silver.py
python pipeline/silver_to_gold.py
```

### 7. Train models

```bash
python models/train_xgboost.py
python models/train_prophet.py
```

### 8. Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📈 Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | — | — | — |
| Prophet | — | — | — |
| Baseline (last value) | — | — | — |

> Metrics will be populated after initial training run. Tracked in MLflow at `http://localhost:5000`.

**Business translation:** a 24h forecast with <8% MAPE allows the city to optimise energy procurement on the spot market, potentially reducing peak-hour costs by 10–15%.

---

## 🔍 MLOps Features

- **Experiment tracking** — every training run logged in MLflow (hyperparameters, metrics, artefacts)
- **Model registry** — champion/challenger versioning with promotion workflow
- **Automated retraining** — GitHub Actions triggers weekly retraining with fresh data
- **Data drift monitoring** — feature distribution shift detection; alerts when model performance degrades
- **Data quality checks** — null rate, range validation, and timestamp continuity enforced at Bronze→Silver transition

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=.
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)

---

<p align="center">
  Built with ❤️ in Zürich · <a href="https://data.stadt-zuerich.ch/">Open Data Zürich</a>
</p>