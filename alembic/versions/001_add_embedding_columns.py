"""Add embedding columns to episodes and skills.

Revision ID: 001
Revises: None
Create Date: 2026-02-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector Vector type only available on Postgres.
    # On SQLite this migration is a no-op for the vector columns.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.add_column("episodes", sa.Column("embedding", sa.Text()))  # will be Vector(1536) with pgvector
        op.add_column("skills", sa.Column("embedding", sa.Text()))
    else:
        # SQLite: add nullable text columns as placeholder
        op.add_column("episodes", sa.Column("embedding", sa.Text(), nullable=True))
        op.add_column("skills", sa.Column("embedding", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("episodes", "embedding")
    op.drop_column("skills", "embedding")
