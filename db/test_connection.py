"""Sanity-check connectivity to DATABASE_URL and list the tables present.

Usage:
    python -m db.test_connection
"""
import logging

import psycopg2

from db.db_client import get_connection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LIST_TABLES_SQL = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
"""


def test_connection() -> None:
    try:
        conn = get_connection()
    except Exception:
        logger.exception("Could not establish a database connection")
        raise

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(LIST_TABLES_SQL)
                rows = cur.fetchall()
        logger.info("Connected successfully.")
        if not rows:
            logger.info("No tables found. Have you run `alembic upgrade head`?")
        else:
            logger.info("Tables found (%d):", len(rows))
            for row in rows:
                print(f"  - {row['table_name']}")
    except psycopg2.Error:
        logger.exception("Query against the database failed")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    test_connection()
