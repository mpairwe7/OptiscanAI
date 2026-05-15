"""additional_seats column on subscriptions

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-15
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column(
            "additional_seats",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "subscriptions",
        sa.Column("stripe_seat_item_id", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("subscriptions", "stripe_seat_item_id")
    op.drop_column("subscriptions", "additional_seats")
