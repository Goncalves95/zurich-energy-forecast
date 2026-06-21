"""add unique constraints to raw_energy and raw_weather

Revision ID: 1b52b4931121
Revises: abed14ccd623
Create Date: 2026-06-21 22:46:23.114282

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '1b52b4931121'
down_revision: Union[str, Sequence[str], None] = 'abed14ccd623'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Required for ON CONFLICT (timestamp, source) DO NOTHING in the
    ingestion scripts (ingestion/fetch_energy.py, ingestion/fetch_weather.py)
    to have a matching constraint to target."""
    op.create_unique_constraint(
        "raw_energy_timestamp_source_unique", "raw_energy", ["timestamp", "source"]
    )
    op.create_unique_constraint(
        "raw_weather_timestamp_source_unique", "raw_weather", ["timestamp", "source"]
    )


def downgrade() -> None:
    op.drop_constraint("raw_weather_timestamp_source_unique", "raw_weather", type_="unique")
    op.drop_constraint("raw_energy_timestamp_source_unique", "raw_energy", type_="unique")
