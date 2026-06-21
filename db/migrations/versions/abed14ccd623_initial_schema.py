"""initial schema

Revision ID: abed14ccd623
Revises:
Create Date: 2026-06-21 22:46:17.776374

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'abed14ccd623'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DOUBLE = postgresql.DOUBLE_PRECISION
TIMESTAMPTZ = sa.DateTime(timezone=True)
NOW = sa.text("NOW()")


def upgrade() -> None:
    """Reflects the Bronze/Silver/Gold schema as it stood in db/schema.sql
    before the raw_energy/raw_weather UNIQUE(timestamp, source) constraints
    were introduced (see the next migration)."""
    op.create_table(
        "raw_energy",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False),
        sa.Column("kwh", DOUBLE()),
        sa.Column("source", sa.Text),
        sa.Column("ingested_at", TIMESTAMPTZ, server_default=NOW),
    )

    op.create_table(
        "raw_weather",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False),
        sa.Column("temperature", DOUBLE()),
        sa.Column("humidity", DOUBLE()),
        sa.Column("solar_rad", DOUBLE()),
        sa.Column("source", sa.Text),
        sa.Column("ingested_at", TIMESTAMPTZ, server_default=NOW),
    )

    op.create_table(
        "clean_energy",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False, unique=True),
        sa.Column("kwh", DOUBLE(), nullable=False),
        sa.Column("processed_at", TIMESTAMPTZ, server_default=NOW),
    )

    op.create_table(
        "clean_weather",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False, unique=True),
        sa.Column("temperature", DOUBLE(), nullable=False),
        sa.Column("humidity", DOUBLE()),
        sa.Column("solar_rad", DOUBLE()),
        sa.Column("processed_at", TIMESTAMPTZ, server_default=NOW),
    )

    op.create_table(
        "features",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False, unique=True),
        sa.Column("kwh", DOUBLE(), nullable=False),
        sa.Column("temperature", DOUBLE()),
        sa.Column("humidity", DOUBLE()),
        sa.Column("solar_rad", DOUBLE()),
        sa.Column("hour", sa.Integer),
        sa.Column("day_of_week", sa.Integer),
        sa.Column("month", sa.Integer),
        sa.Column("is_weekend", sa.Boolean),
        sa.Column("is_holiday", sa.Boolean),
        sa.Column("lag_1h", DOUBLE()),
        sa.Column("lag_24h", DOUBLE()),
        sa.Column("rolling_avg_7d", DOUBLE()),
        sa.Column("created_at", TIMESTAMPTZ, server_default=NOW),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", TIMESTAMPTZ, nullable=False),
        sa.Column("model_name", sa.Text, nullable=False),
        sa.Column("model_version", sa.Text),
        sa.Column("predicted_kwh", DOUBLE(), nullable=False),
        sa.Column("created_at", TIMESTAMPTZ, server_default=NOW),
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("features")
    op.drop_table("clean_weather")
    op.drop_table("clean_energy")
    op.drop_table("raw_weather")
    op.drop_table("raw_energy")
