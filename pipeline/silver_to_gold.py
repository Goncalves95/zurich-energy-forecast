"""Transform clean (Silver) data into ML-ready features (Gold layer)."""
import logging

import holidays
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from db.db_client import get_connection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INSERT_BATCH_SIZE = 1000
ROLLING_WINDOW = "7D"

# Calendar features (hour/day_of_week/month/is_holiday) are computed in local
# Zürich time, since that's what governs human consumption behaviour, not UTC.
LOCAL_TIMEZONE = "Europe/Zurich"
HOLIDAY_SUBDIVISION = "ZH"

FEATURE_COLUMNS = [
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
UPDATE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "timestamp"]


def _read_table(conn, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def build_features(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join clean_energy with clean_weather on timestamp and engineer ML features.

    Energy rows are always kept even when there's no matching weather reading
    (weather columns are NULL for those rows): MeteoSwiss's "messwerte-aktuell"
    source only ever exposes the latest reading, so weather history is thin
    until it accumulates via repeated scheduler runs.
    """
    if energy.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    if weather.empty:
        df = energy.copy()
        for column in ("temperature", "humidity", "solar_rad"):
            df[column] = pd.NA
        n_unmatched = len(df)
    else:
        df = energy.merge(weather, on="timestamp", how="left")
        n_unmatched = int(df["temperature"].isna().sum())

    if n_unmatched:
        logger.warning(
            "silver_to_gold: %d/%d energy rows have no matching weather reading",
            n_unmatched, len(df),
        )

    df = df.sort_values("timestamp").set_index("timestamp")

    local_index = df.index.tz_convert(LOCAL_TIMEZONE)
    local_dates = local_index.date

    df["hour"] = local_index.hour
    df["day_of_week"] = local_index.dayofweek
    df["month"] = local_index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    years = sorted({d.year for d in local_dates})
    ch_holidays = holidays.CH(subdiv=HOLIDAY_SUBDIVISION, years=years)
    df["is_holiday"] = [d in ch_holidays for d in local_dates]

    # Exact elapsed-time lookups: NaN if that timestamp isn't present (e.g. a gap).
    df["lag_1h"] = df["kwh"].reindex(df.index - pd.Timedelta(hours=1)).to_numpy()
    df["lag_24h"] = df["kwh"].reindex(df.index - pd.Timedelta(hours=24)).to_numpy()

    df["rolling_avg_7d"] = df["kwh"].rolling(ROLLING_WINDOW, min_periods=1).mean()

    df = df.reset_index()
    return df[FEATURE_COLUMNS]


def insert_features(conn, df: pd.DataFrame) -> tuple[int, int]:
    """Upsert feature rows. Returns (inserted, refreshed)."""
    total = len(df)
    if df.empty:
        return 0, 0

    timestamps = df["timestamp"].tolist()
    with conn.cursor() as cur:
        cur.execute("SELECT timestamp FROM features WHERE timestamp = ANY(%s)", (timestamps,))
        existing = {row["timestamp"] for row in cur.fetchall()}

    def _clean(value):
        return None if pd.isna(value) else value

    rows = [
        tuple(_clean(v) for v in row)
        for row in df[FEATURE_COLUMNS].itertuples(index=False, name=None)
    ]
    set_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in UPDATE_COLUMNS)

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO features ({", ".join(FEATURE_COLUMNS)})
            VALUES %s
            ON CONFLICT (timestamp) DO UPDATE SET {set_clause}
            """,
            rows,
            page_size=INSERT_BATCH_SIZE,
        )
    conn.commit()

    refreshed = len(existing)
    return total - refreshed, refreshed


def run() -> None:
    """Read clean_energy/clean_weather, engineer features, and upsert into features."""
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        energy = _read_table(conn, "SELECT timestamp, kwh FROM clean_energy")
        weather = _read_table(
            conn, "SELECT timestamp, temperature, humidity, solar_rad FROM clean_weather"
        )

        features = build_features(energy, weather)
        logger.info(
            "silver_to_gold: %d clean_energy rows, %d clean_weather rows -> %d joined feature rows",
            len(energy), len(weather), len(features),
        )

        inserted, refreshed = insert_features(conn, features)
        logger.info("features: %d rows inserted, %d rows refreshed (upserted)", inserted, refreshed)
    except psycopg2.Error:
        conn.rollback()
        logger.exception("Database error while running silver_to_gold")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
