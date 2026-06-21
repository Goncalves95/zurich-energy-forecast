"""Fetch electricity consumption data for Zürich (ewz) from Open Data Zürich.

Data source: CKAN package `ewz_stromabgabe_netzebenen_stadt_zuerich`
(15-minute electricity consumption across network levels NE5/NE7,
Stadt Zürich, since 2015).
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

CKAN_PACKAGE_SHOW_URL = "https://data.stadt-zuerich.ch/api/3/action/package_show"
PACKAGE_ID = "ewz_stromabgabe_netzebenen_stadt_zuerich"
SOURCE_NAME = "ewz_stromabgabe_netzebenen_stadt_zuerich"
REQUEST_TIMEOUT = 60
INCREMENTAL_LOOKBACK_DAYS = 7


def discover_resource_url(package_id: str = PACKAGE_ID) -> str:
    """Ask the CKAN API for the package and return its CSV resource URL."""
    try:
        response = requests.get(
            CKAN_PACKAGE_SHOW_URL, params={"id": package_id}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        logger.exception("Failed to query CKAN package_show for id=%s", package_id)
        raise

    payload = response.json()
    resources = payload.get("result", {}).get("resources", [])
    for resource in resources:
        if resource.get("format", "").upper() == "CSV" and resource.get("url"):
            return resource["url"]

    raise RuntimeError(f"No CSV resource found in CKAN package '{package_id}'")


def download_csv(url: str) -> str:
    """Download the CSV resource and return its raw text content."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        logger.exception("Failed to download CSV resource from %s", url)
        raise
    return response.text


def _parse_timestamp(value):
    """Parse a timestamp, falling back from ISO 8601 to Swiss dd.mm.yyyy format."""
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, utc=True, errors="coerce", dayfirst=True)
    return parsed


def parse_records(csv_text: str) -> list[tuple]:
    """Parse raw CSV text into (timestamp, kwh) tuples, skipping malformed rows."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        logger.exception("Failed to parse CSV content")
        raise

    if df.empty:
        logger.warning("Downloaded CSV has no rows")
        return []

    df["timestamp"] = df["Timestamp"].apply(_parse_timestamp)
    df["Value_NE5"] = pd.to_numeric(df.get("Value_NE5"), errors="coerce")
    df["Value_NE7"] = pd.to_numeric(df.get("Value_NE7"), errors="coerce")

    malformed = df["timestamp"].isna() | (df["Value_NE5"].isna() & df["Value_NE7"].isna())
    n_malformed = int(malformed.sum())
    if n_malformed:
        logger.warning("Skipping %d malformed/missing rows", n_malformed)
    df = df[~malformed]

    # Total city consumption = sum of the two network levels (NE5 + NE7).
    df["kwh"] = df["Value_NE5"].fillna(0) + df["Value_NE7"].fillna(0)

    return list(zip(df["timestamp"].tolist(), df["kwh"].tolist()))


def _last_ingested_timestamp(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(timestamp) AS max_ts FROM raw_energy WHERE source = %s",
            (SOURCE_NAME,),
        )
        row = cur.fetchone()
    return row["max_ts"] if row else None


INSERT_BATCH_SIZE = 1000


def insert_records(conn, records: list[tuple]) -> tuple[int, int]:
    """Bulk insert (timestamp, kwh) records in batches, skipping duplicates.

    Returns (inserted, skipped). `RETURNING id` is used to count actual
    insertions, since ON CONFLICT DO NOTHING makes cursor.rowcount unreliable
    across the multiple batched statements execute_values issues.
    """
    total = len(records)
    rows = [(timestamp, kwh, SOURCE_NAME) for timestamp, kwh in records]

    with conn.cursor() as cur:
        returned = execute_values(
            cur,
            """
            INSERT INTO raw_energy (timestamp, kwh, source)
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


def fetch_energy(full_load: bool = False) -> None:
    """Fetch ewz electricity consumption data and load it into raw_energy."""
    resource_url = discover_resource_url()
    logger.info("Discovered CSV resource: %s", resource_url)

    csv_text = download_csv(resource_url)
    records = parse_records(csv_text)
    logger.info("Parsed %d valid records from CSV", len(records))

    if not records:
        logger.info("Nothing to insert")
        return

    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        if not full_load:
            last_ts = _last_ingested_timestamp(conn)
            if last_ts is not None:
                records = [r for r in records if r[0] > last_ts]
                logger.info("Incremental load: %d records newer than %s", len(records), last_ts)
            else:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
                records = [r for r in records if r[0] > cutoff]
                logger.info(
                    "No prior data found; limiting to the last %d days (%d records). "
                    "Use --full-load to ingest the entire history.",
                    INCREMENTAL_LOOKBACK_DAYS,
                    len(records),
                )

        if not records:
            logger.info("No new records to insert")
            return

        inserted, skipped = insert_records(conn, records)
        logger.info("Rows inserted: %d, rows skipped (duplicates): %d", inserted, skipped)
    except psycopg2.Error:
        conn.rollback()
        logger.exception("Database error while inserting raw_energy records")
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ewz electricity consumption data")
    parser.add_argument(
        "--full-load",
        action="store_true",
        help="Ingest the entire historical series instead of only new records",
    )
    args = parser.parse_args()
    fetch_energy(full_load=args.full_load)


if __name__ == "__main__":
    main()
