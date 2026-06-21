"""PostgreSQL connection and query helpers."""
import logging
import os

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


def get_connection():
    """Open a connection using the DATABASE_URL env var.

    Expected format (Neon.tech or any standard Postgres URL):
        postgresql://user:password@host/dbname?sslmode=require
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file, e.g. "
            "postgresql://user:password@host/dbname?sslmode=require"
        )

    try:
        return psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    except psycopg2.OperationalError:
        logger.exception("Failed to connect to the database at %s", _redact(database_url))
        raise


def _redact(database_url: str) -> str:
    """Strip credentials from a connection string before logging it."""
    if "@" not in database_url:
        return database_url
    scheme_and_creds, _, host_and_rest = database_url.partition("@")
    scheme = scheme_and_creds.split("://", 1)[0]
    return f"{scheme}://***@{host_and_rest}"
