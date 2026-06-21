"""Fetch hourly weather data for Zürich (station SMA / Fluntern) from MeteoSwiss open data.

Data source: "messwerte-aktuell" CSV — current measurements across all Swiss
stations, refreshed every ~10 minutes. NOTE: this endpoint only ever exposes
the latest reading per station; it has no historical archive. The --full-load
flag therefore widens the insert cutoff rather than performing a real 30-day
backfill — to build up hourly history you need to run this repeatedly (e.g.
on a scheduler) so each run's snapshot accumulates in raw_weather.
"""
import argparse
import io
import logging

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from db.db_client import get_connection

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_URL = "https://data.geo.admin.ch/ch.meteoschweiz.messwerte-aktuell/VQHA80.csv"
STATION_CODE = "SMA"
STATION_COLUMN = "Station/Location"
TIMESTAMP_COLUMN = "Date"
TEMPERATURE_COLUMN = "tre200s0"
HUMIDITY_COLUMN = "ure200s0"
SOLAR_RAD_COLUMN = "gre000z0"
# The "Date" column is local Swiss time (no UTC offset in the raw value).
STATION_TIMEZONE = "Europe/Zurich"

SOURCE_NAME = "meteoswiss_messwerte_aktuell"
REQUEST_TIMEOUT = 60
INSERT_BATCH_SIZE = 1000
INCREMENTAL_LOOKBACK_DAYS = 1
FULL_LOAD_LOOKBACK_DAYS = 30


def download_csv(url: str = CSV_URL) -> str:
    """Download the MeteoSwiss CSV and return its raw text content."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        logger.exception("Failed to download MeteoSwiss CSV from %s", url)
        raise
    return response.text


def _clean(value):
    """Convert a numeric value to a plain float, or None for missing/NaN."""
    return None if pd.isna(value) else float(value)


def parse_records(csv_text: str, station_code: str = STATION_CODE) -> list[tuple]:
    """Parse MeteoSwiss CSV text into (timestamp, temperature, humidity, solar_rad)
    tuples for the given station, skipping malformed rows."""
    try:
        df = pd.read_csv(io.StringIO(csv_text), sep=";", na_values=["-"])
    except Exception:
        logger.exception("Failed to parse MeteoSwiss CSV content")
        raise

    if df.empty:
        logger.warning("Downloaded MeteoSwiss CSV has no rows")
        return []

    required_columns = {
        STATION_COLUMN,
        TIMESTAMP_COLUMN,
        TEMPERATURE_COLUMN,
        HUMIDITY_COLUMN,
        SOLAR_RAD_COLUMN,
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error("MeteoSwiss CSV is missing expected columns: %s", sorted(missing_columns))
        return []

    df = df[df[STATION_COLUMN] == station_code].copy()
    if df.empty:
        logger.warning("No rows found for station '%s'", station_code)
        return []

    timestamp = pd.to_datetime(df[TIMESTAMP_COLUMN], format="%Y%m%d%H%M", errors="coerce")
    df["timestamp"] = (
        timestamp.dt.tz_localize(STATION_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )

    for column in (TEMPERATURE_COLUMN, HUMIDITY_COLUMN, SOLAR_RAD_COLUMN):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    malformed = df["timestamp"].isna()
    n_malformed = int(malformed.sum())
    if n_malformed:
        logger.warning("Skipping %d row(s) with an unparseable timestamp", n_malformed)
    df = df[~malformed]

    return [
        (ts, _clean(temp), _clean(hum), _clean(solar))
        for ts, temp, hum, solar in zip(
            df["timestamp"].tolist(),
            df[TEMPERATURE_COLUMN].tolist(),
            df[HUMIDITY_COLUMN].tolist(),
            df[SOLAR_RAD_COLUMN].tolist(),
        )
    ]


def insert_records(conn, records: list[tuple]) -> tuple[int, int]:
    """Bulk insert (timestamp, temperature, humidity, solar_rad) records in batches,
    skipping duplicates. Returns (inserted, skipped)."""
    total = len(records)
    rows = [(ts, temp, hum, solar, SOURCE_NAME) for ts, temp, hum, solar in records]

    with conn.cursor() as cur:
        returned = execute_values(
            cur,
            """
            INSERT INTO raw_weather (timestamp, temperature, humidity, solar_rad, source)
            VALUES %s
            ON CONFLICT (timestamp, source) DO NOTHING
            RETURNING id
            """,
            rows,
            page_size=INSERT_BATCH_SIZE,
            fetch=True,
        )
    conn.commit()

    inserted = len(returned)
    skipped = total - inserted
    return inserted, skipped


def fetch_weather(full_load: bool = False) -> None:
    """Fetch the current MeteoSwiss reading for Zürich and load it into raw_weather."""
    csv_text = download_csv()
    records = parse_records(csv_text)
    logger.info("Parsed %d valid record(s) for station '%s'", len(records), STATION_CODE)

    if not records:
        logger.info("Nothing to insert")
        return

    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        lookback_days = FULL_LOAD_LOOKBACK_DAYS if full_load else INCREMENTAL_LOOKBACK_DAYS
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
        records = [r for r in records if r[0] > cutoff]

        if not records:
            logger.info("No records newer than cutoff %s to insert", cutoff)
            return

        inserted, skipped = insert_records(conn, records)
        logger.info("Rows inserted: %d, rows skipped (duplicates): %d", inserted, skipped)
    except psycopg2.Error:
        conn.rollback()
        logger.exception("Database error while inserting raw_weather records")
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MeteoSwiss weather data for Zürich (SMA)")
    parser.add_argument(
        "--full-load",
        action="store_true",
        help="Use a 30-day lookback cutoff instead of the default 1-day incremental cutoff",
    )
    args = parser.parse_args()
    fetch_weather(full_load=args.full_load)


if __name__ == "__main__":
    main()
