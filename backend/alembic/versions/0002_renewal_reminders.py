"""renewal_reminders table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-15
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    sa.Enum(
        "7d", "3d", "1d", "expired", name="renewal_reminder_kind",
    ).create(op.get_bind(), checkfirst=True)

    reminder_kind = postgresql.ENUM(name="renewal_reminder_kind", create_type=False)

    op.create_table(
        "renewal_reminders",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "subscription_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("subscriptions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", reminder_kind, nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_to", sa.String(320), nullable=False),
        sa.Column(
            "sent_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("error", sa.String(2000), nullable=True),
        sa.UniqueConstraint(
            "subscription_id",
            "period_end",
            "kind",
            name="uq_renewal_reminder_sub_period_kind",
        ),
    )
    op.create_index(
        "ix_renewal_reminders_subscription_id",
        "renewal_reminders",
        ["subscription_id"],
    )


def downgrade() -> None:
    op.drop_table("renewal_reminders")
    sa.Enum(name="renewal_reminder_kind").drop(op.get_bind(), checkfirst=True)
