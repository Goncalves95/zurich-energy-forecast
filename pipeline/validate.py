"""Data quality checks at each layer transition."""
import logging

import pandas as pd
from dotenv import load_dotenv

from db.db_client import get_connection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BRONZE_NULL_RATE_THRESHOLD = 0.05
GOLD_NULL_RATE_THRESHOLD = 0.01
# Weather coverage accumulates gradually via the scheduler (see the LEFT JOIN
# in silver_to_gold.py), so these columns get a much looser gate than the
# energy/calendar features, which should always be fully populated.
GOLD_WEATHER_NULL_RATE_THRESHOLD = 0.80
GOLD_WEATHER_COLUMNS = ["temperature", "humidity", "solar_rad"]

GOLD_FEATURE_COLUMNS = [
    "timestamp",
    "kwh",
    "temperature",
    "humidity",
    "solar_rad",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1h",
    "lag_24h",
    "rolling_avg_7d",
]


def _read_table(conn, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def _null_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())


def validate_bronze() -> bool:
    """Bronze gate: raw_energy's timestamp and kwh null rates must stay under 5%."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        df = _read_table(conn, "SELECT timestamp, kwh FROM raw_energy")
    finally:
        conn.close()

    if df.empty:
        logger.warning("validate_bronze: raw_energy is empty")
        return True

    passed = True
    for column in ("timestamp", "kwh"):
        rate = _null_rate(df[column])
        if rate >= BRONZE_NULL_RATE_THRESHOLD:
            logger.error(
                "validate_bronze: raw_energy.%s null rate %.2f%% >= %.0f%% threshold",
                column, rate * 100, BRONZE_NULL_RATE_THRESHOLD * 100,
            )
            passed = False
    return passed


def validate_silver() -> bool:
    """Silver gate: no negative kwh values and no duplicate timestamps
    in clean_energy or clean_weather."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        energy = _read_table(conn, "SELECT timestamp, kwh FROM clean_energy")
        weather = _read_table(conn, "SELECT timestamp FROM clean_weather")
    finally:
        conn.close()

    passed = True

    if not energy.empty:
        n_negative = int((energy["kwh"] < 0).sum())
        if n_negative:
            logger.error("validate_silver: clean_energy has %d negative kwh value(s)", n_negative)
            passed = False

        n_dupes = int(energy["timestamp"].duplicated().sum())
        if n_dupes:
            logger.error("validate_silver: clean_energy has %d duplicate timestamp(s)", n_dupes)
            passed = False

    if not weather.empty:
        n_dupes = int(weather["timestamp"].duplicated().sum())
        if n_dupes:
            logger.error("validate_silver: clean_weather has %d duplicate timestamp(s)", n_dupes)
            passed = False

    return passed


def validate_gold() -> bool:
    """Gold gate: features table must have every expected column. Energy/calendar
    columns must stay under a 1% null rate; weather columns get a looser 80%
    null rate threshold while history accumulates via the scheduler."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        df = _read_table(conn, "SELECT * FROM features")
    finally:
        conn.close()

    missing_columns = [c for c in GOLD_FEATURE_COLUMNS if c not in df.columns]
    if missing_columns:
        logger.error("validate_gold: features table is missing columns: %s", missing_columns)
        return False

    if df.empty:
        logger.warning("validate_gold: features table is empty")
        return True

    passed = True
    for column in GOLD_FEATURE_COLUMNS:
        threshold = (
            GOLD_WEATHER_NULL_RATE_THRESHOLD
            if column in GOLD_WEATHER_COLUMNS
            else GOLD_NULL_RATE_THRESHOLD
        )
        rate = _null_rate(df[column])
        if rate >= threshold:
            logger.error(
                "validate_gold: features.%s null rate %.2f%% >= %.0f%% threshold",
                column, rate * 100, threshold * 100,
            )
            passed = False
    return passed
