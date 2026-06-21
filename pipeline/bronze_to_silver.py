"""Transform raw (Bronze) data into cleaned (Silver) layer."""
import logging

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from db.db_client import get_connection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INSERT_BATCH_SIZE = 1000

# Valid kwh range is (0, 180000]. raw_energy.kwh is whole-city consumption
# (NE5 + NE7 network levels) per 15-minute interval, not a per-household
# reading, so the historical range is far above a "normal" kWh figure:
# observed min/max over 402k rows is ~16.9k/~122.3k, median ~75.7k, p99.9
# ~115.7k. 180000 is the 3*IQR Tukey upper fence (Q3=91.0k, IQR=29.8k),
# giving headroom above the observed max while still catching genuine
# sensor/unit-conversion faults. The old 50000 cutoff was below the median
# and was silently dropping roughly half of all readings as "outliers".
ENERGY_KWH_MIN = 0
ENERGY_KWH_MAX = 180_000
ENERGY_FFILL_LIMIT_HOURS = 3

WEATHER_TEMP_MIN = -30.0
WEATHER_TEMP_MAX = 45.0
WEATHER_HUMIDITY_MIN = 0.0
WEATHER_HUMIDITY_MAX = 100.0


def _read_table(conn, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def clean_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, drop outliers, and forward-fill short gaps in raw energy readings.

    Raw readings come in on a 15-minute grid; they're resampled to hourly
    (mean) here since the downstream Gold features (lag_1h, lag_24h, hour)
    are all defined on an hourly cadence.
    """
    total = len(df)
    if df.empty:
        logger.info("clean_energy: no input rows")
        return df.assign(kwh=pd.Series(dtype="float64")) if "kwh" in df.columns else df

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    n_dupes = total - len(df)

    in_range = df["kwh"].notna() & (df["kwh"] > ENERGY_KWH_MIN) & (df["kwh"] <= ENERGY_KWH_MAX)
    n_outliers = int((~in_range).sum())
    df = df[in_range].copy()

    if df.empty:
        logger.info(
            "clean_energy: %d input rows -> 0 output rows "
            "(%d duplicates removed, %d outliers dropped)",
            total, n_dupes, n_outliers,
        )
        return df

    hourly = df.set_index("timestamp")["kwh"].resample("1h").mean()
    hourly = hourly.ffill(limit=ENERGY_FFILL_LIMIT_HOURS)
    n_unfilled_gaps = int(hourly.isna().sum())
    hourly = hourly.dropna()

    result = hourly.reset_index()

    logger.info(
        "clean_energy: %d input rows -> %d output rows (%d duplicates removed, "
        "%d outliers dropped, %d hourly gaps left unfilled beyond %dh limit)",
        total, len(result), n_dupes, n_outliers, n_unfilled_gaps, ENERGY_FFILL_LIMIT_HOURS,
    )
    return result[["timestamp", "kwh"]]


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, clamp sensor ranges, and drop unusable rows in raw weather readings."""
    total = len(df)
    if df.empty:
        logger.info("clean_weather: no input rows")
        return df

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last").copy()
    n_dupes = total - len(df)

    df["temperature"] = df["temperature"].clip(lower=WEATHER_TEMP_MIN, upper=WEATHER_TEMP_MAX)
    df["humidity"] = df["humidity"].clip(lower=WEATHER_HUMIDITY_MIN, upper=WEATHER_HUMIDITY_MAX)

    all_null = df[["temperature", "humidity", "solar_rad"]].isna().all(axis=1)
    n_all_null = int(all_null.sum())
    df = df[~all_null]

    # clean_weather.temperature is NOT NULL in the schema, so rows that survive
    # the all-null check but still lack a temperature reading can't be kept.
    missing_temp = df["temperature"].isna()
    n_missing_temp = int(missing_temp.sum())
    df = df[~missing_temp]

    logger.info(
        "clean_weather: %d input rows -> %d output rows "
        "(%d duplicates removed, %d all-null rows dropped, %d missing-temperature rows dropped)",
        total, len(df), n_dupes, n_all_null, n_missing_temp,
    )
    return df[["timestamp", "temperature", "humidity", "solar_rad"]]


def insert_clean_energy(conn, df: pd.DataFrame) -> tuple[int, int]:
    """Bulk insert cleaned energy rows. Returns (inserted, skipped)."""
    total = len(df)
    if df.empty:
        return 0, 0

    rows = list(df[["timestamp", "kwh"]].itertuples(index=False, name=None))
    with conn.cursor() as cur:
        returned = execute_values(
            cur,
            """
            INSERT INTO clean_energy (timestamp, kwh)
            VALUES %s
            ON CONFLICT (timestamp) DO NOTHING
            RETURNING id
            """,
            rows,
            page_size=INSERT_BATCH_SIZE,
            fetch=True,
        )
    conn.commit()

    inserted = len(returned)
    return inserted, total - inserted


def insert_clean_weather(conn, df: pd.DataFrame) -> tuple[int, int]:
    """Bulk insert cleaned weather rows. Returns (inserted, skipped)."""
    total = len(df)
    if df.empty:
        return 0, 0

    def _clean(value):
        return None if pd.isna(value) else float(value)

    columns = df[["timestamp", "temperature", "humidity", "solar_rad"]]
    rows = [
        (ts, float(temp), _clean(hum), _clean(solar))
        for ts, temp, hum, solar in columns.itertuples(index=False, name=None)
    ]

    with conn.cursor() as cur:
        returned = execute_values(
            cur,
            """
            INSERT INTO clean_weather (timestamp, temperature, humidity, solar_rad)
            VALUES %s
            ON CONFLICT (timestamp) DO NOTHING
            RETURNING id
            """,
            rows,
            page_size=INSERT_BATCH_SIZE,
            fetch=True,
        )
    conn.commit()

    inserted = len(returned)
    return inserted, total - inserted


def run() -> None:
    """Read raw_energy/raw_weather, clean them, and load clean_energy/clean_weather."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        raw_energy = _read_table(conn, "SELECT timestamp, kwh FROM raw_energy")
        raw_weather = _read_table(
            conn, "SELECT timestamp, temperature, humidity, solar_rad FROM raw_weather"
        )

        cleaned_energy = clean_energy(raw_energy)
        cleaned_weather = clean_weather(raw_weather)

        inserted_e, skipped_e = insert_clean_energy(conn, cleaned_energy)
        logger.info(
            "clean_energy: %d rows inserted, %d rows skipped (duplicates)", inserted_e, skipped_e
        )

        inserted_w, skipped_w = insert_clean_weather(conn, cleaned_weather)
        logger.info(
            "clean_weather: %d rows inserted, %d rows skipped (duplicates)", inserted_w, skipped_w
        )
    except psycopg2.Error:
        conn.rollback()
        logger.exception("Database error while running bronze_to_silver")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
