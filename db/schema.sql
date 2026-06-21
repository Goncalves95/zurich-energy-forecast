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
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (timestamp, source)
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
