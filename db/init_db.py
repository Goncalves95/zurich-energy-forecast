"""Apply schema.sql to the database pointed to by DATABASE_URL.

Usage:
    python -m db.init_db
"""
import logging
from pathlib import Path

import psycopg2

from db.db_client import get_connection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_db() -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        logger.info("Schema applied successfully from %s", SCHEMA_PATH.name)
    except psycopg2.Error:
        logger.exception("Failed to apply schema")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
