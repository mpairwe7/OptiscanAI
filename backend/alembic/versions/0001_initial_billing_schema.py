"""initial billing schema + plan seed

Revision ID: 0001
Revises:
Create Date: 2026-05-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enums ──
    # Create types explicitly, then use `create_type=False` everywhere they're
    # referenced as column types — prevents the double-CREATE TYPE that would
    # otherwise occur on the first column referencing each enum.
    sa.Enum(
        "owner",
        "admin",
        "clinician",
        "viewer",
        name="membership_role",
    ).create(op.get_bind(), checkfirst=True)
    sa.Enum(
        "pending",
        "active",
        "revoked",
        name="membership_status",
    ).create(op.get_bind(), checkfirst=True)
    sa.Enum(
        "trialing",
        "active",
        "past_due",
        "canceled",
        "incomplete",
        name="subscription_status",
    ).create(op.get_bind(), checkfirst=True)
    sa.Enum("monthly", "annual", name="billing_cycle").create(op.get_bind(), checkfirst=True)
    sa.Enum(
        "stripe",
        "mtn",
        "airtel",
        "flutterwave",
        "manual",
        name="payment_provider",
    ).create(op.get_bind(), checkfirst=True)
    sa.Enum(
        "draft",
        "open",
        "paid",
        "void",
        "uncollectible",
        name="invoice_status",
    ).create(op.get_bind(), checkfirst=True)
    sa.Enum(
        "requires_action",
        "processing",
        "succeeded",
        "failed",
        "canceled",
        name="payment_intent_status",
    ).create(op.get_bind(), checkfirst=True)
    sa.Enum(
        "scan",
        "explain_gradcam",
        "explain_lime",
        "explain_shap",
        "explain_ig",
        "explain_eli5",
        "clinical_reasoning",
        "audit_export",
        name="usage_event_type",
    ).create(op.get_bind(), checkfirst=True)

    # Column-side references — `create_type=False` skips the auto-create that
    # would otherwise re-attempt CREATE TYPE on first column use.
    membership_role = postgresql.ENUM(name="membership_role", create_type=False)
    membership_status = postgresql.ENUM(name="membership_status", create_type=False)
    subscription_status = postgresql.ENUM(name="subscription_status", create_type=False)
    billing_cycle = postgresql.ENUM(name="billing_cycle", create_type=False)
    payment_provider = postgresql.ENUM(name="payment_provider", create_type=False)
    invoice_status = postgresql.ENUM(name="invoice_status", create_type=False)
    payment_intent_status = postgresql.ENUM(name="payment_intent_status", create_type=False)
    usage_event_type = postgresql.ENUM(name="usage_event_type", create_type=False)

    # ── users ──
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("email_normalized", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(512), nullable=True),
        sa.Column("full_name", sa.String(200), nullable=True),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("is_superuser", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("practitioner_role", sa.String(50), nullable=True),
        sa.Column("country", sa.String(2), nullable=True),
        sa.Column("facility_name", sa.String(200), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email_normalized", "users", ["email_normalized"])
    op.create_index("ix_users_email_normalized_active", "users", ["email_normalized", "is_active"])

    # ── plans ──
    op.create_table(
        "plans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("code", sa.String(32), nullable=False, unique=True),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("tagline", sa.String(200), nullable=True),
        sa.Column("monthly_price_cents", sa.Integer, nullable=True),
        sa.Column("annual_price_cents", sa.Integer, nullable=True),
        sa.Column("currency", sa.String(3), nullable=False, server_default="USD"),
        sa.Column("scan_limit_monthly", sa.Integer, nullable=True),
        sa.Column("seat_limit", sa.Integer, nullable=True),
        sa.Column("is_contact_sales", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("is_featured", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("features", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("stripe_price_id_monthly", sa.String(100), nullable=True),
        sa.Column("stripe_price_id_annual", sa.String(100), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_plans_code", "plans", ["code"])

    # ── organizations ──
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("billing_email", sa.String(320), nullable=True),
        sa.Column(
            "owner_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("is_personal", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_organizations_slug", "organizations", ["slug"])

    # ── memberships ──
    op.create_table(
        "memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", membership_role, nullable=False, server_default="clinician"),
        sa.Column("status", membership_status, nullable=False, server_default="active"),
        sa.Column(
            "invited_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("invited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
    )
    op.create_index("ix_memberships_user_id", "memberships", ["user_id"])
    op.create_index("ix_memberships_organization_id", "memberships", ["organization_id"])
    op.create_index("ix_memberships_status", "memberships", ["status"])

    # ── subscriptions ──
    op.create_table(
        "subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "plan_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", subscription_status, nullable=False, server_default="active"),
        sa.Column("billing_cycle", billing_cycle, nullable=False, server_default="monthly"),
        sa.Column("provider", payment_provider, nullable=False, server_default="manual"),
        sa.Column(
            "current_period_start",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cancel_at_period_end", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("canceled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trial_ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(100), nullable=True),
        sa.Column("stripe_customer_id", sa.String(100), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("organization_id", name="uq_subscription_organization"),
    )
    op.create_index(
        "ix_subscription_status_period_end", "subscriptions", ["status", "current_period_end"]
    )

    # ── invoices ──
    op.create_table(
        "invoices",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "subscription_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("subscriptions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("amount_cents", sa.Integer, nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="USD"),
        sa.Column("status", invoice_status, nullable=False, server_default="draft"),
        sa.Column("provider", payment_provider, nullable=False),
        sa.Column("provider_invoice_id", sa.String(100), nullable=True),
        sa.Column("hosted_url", sa.String(500), nullable=True),
        sa.Column("pdf_url", sa.String(500), nullable=True),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_invoices_organization_id", "invoices", ["organization_id"])
    op.create_index("ix_invoice_org_issued", "invoices", ["organization_id", "issued_at"])
    op.create_index("ix_invoices_provider_invoice_id", "invoices", ["provider_invoice_id"])

    # ── payment_intents ──
    op.create_table(
        "payment_intents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "subscription_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("subscriptions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "invoice_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("invoices.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("provider", payment_provider, nullable=False),
        sa.Column("provider_intent_id", sa.String(200), nullable=True, unique=True),
        sa.Column("idempotency_key", sa.String(100), nullable=False, unique=True),
        sa.Column("amount_cents", sa.Integer, nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="USD"),
        sa.Column(
            "status", payment_intent_status, nullable=False, server_default="requires_action"
        ),
        sa.Column("phone_msisdn", sa.String(20), nullable=True),
        sa.Column("plan_code", sa.String(32), nullable=True),
        sa.Column("billing_cycle", sa.String(20), nullable=True),
        sa.Column("raw_callback", sa.JSON, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_payment_intents_organization_id", "payment_intents", ["organization_id"])

    # ── usage_events ──
    op.create_table(
        "usage_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("event_type", usage_event_type, nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False, server_default="1"),
        sa.Column("request_id", sa.String(50), nullable=True),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_usage_org_type_occurred",
        "usage_events",
        ["organization_id", "event_type", "occurred_at"],
    )
    op.create_index("ix_usage_occurred", "usage_events", ["occurred_at"])

    # ── webhook_events ──
    op.create_table(
        "webhook_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("provider", payment_provider, nullable=False),
        sa.Column("provider_event_id", sa.String(200), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=True),
        sa.Column("payload", sa.JSON, nullable=True),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.String(2000), nullable=True),
        sa.UniqueConstraint("provider", "provider_event_id", name="uq_webhook_provider_event"),
    )

    # ── refresh_tokens ──
    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_fingerprint", sa.String(200), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_user_revoked", "refresh_tokens", ["user_id", "revoked_at"])

    # ── email_verification_tokens ──
    op.create_table(
        "email_verification_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_email_verification_tokens_user_id", "email_verification_tokens", ["user_id"]
    )

    # ── password_reset_tokens ──
    op.create_table(
        "password_reset_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"])

    # ── magic_link_tokens ──
    op.create_table(
        "magic_link_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("token_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_magic_link_tokens_email", "magic_link_tokens", ["email"])

    # ── organization_invites ──
    op.create_table(
        "organization_invites",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "invited_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("role", membership_role, nullable=False, server_default="clinician"),
        sa.Column("token_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_organization_invites_organization_id", "organization_invites", ["organization_id"]
    )
    op.create_index("ix_organization_invites_email", "organization_invites", ["email"])

    # ── Seed plans ──
    _seed_plans()


def _seed_plans() -> None:
    import uuid

    plans_table = sa.table(
        "plans",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("code", sa.String),
        sa.column("display_name", sa.String),
        sa.column("description", sa.String),
        sa.column("tagline", sa.String),
        sa.column("monthly_price_cents", sa.Integer),
        sa.column("annual_price_cents", sa.Integer),
        sa.column("currency", sa.String),
        sa.column("scan_limit_monthly", sa.Integer),
        sa.column("seat_limit", sa.Integer),
        sa.column("is_contact_sales", sa.Boolean),
        sa.column("is_featured", sa.Boolean),
        sa.column("sort_order", sa.Integer),
        sa.column("features", sa.JSON),
    )

    op.bulk_insert(
        plans_table,
        [
            {
                "id": uuid.uuid4(),
                "code": "free",
                "display_name": "Free",
                "tagline": "Try clinical screening with no commitment.",
                "description": "Get started with 10 retinal scans per month and core Grad-CAM explainability.",
                "monthly_price_cents": 0,
                "annual_price_cents": 0,
                "currency": "USD",
                "scan_limit_monthly": 10,
                "seat_limit": 1,
                "is_contact_sales": False,
                "is_featured": False,
                "sort_order": 0,
                "features": {
                    "scans_per_month": "10",
                    "diseases": "45-disease detection",
                    "explainability": "Grad-CAM",
                    "report_retention_days": 7,
                    "watermark": True,
                    "support": "Community",
                    "audit_log": False,
                    "team_seats": False,
                    "dhis2_fhir": False,
                    "sso": False,
                },
            },
            {
                "id": uuid.uuid4(),
                "code": "clinician",
                "display_name": "Clinician",
                "tagline": "For solo ophthalmologists and optometrists.",
                "description": "500 scans/month, every explainability method, clinical reasoning, PDF reports.",
                "monthly_price_cents": 2900,
                "annual_price_cents": 29000,  # $290/yr (17% off $348)
                "currency": "USD",
                "scan_limit_monthly": 500,
                "seat_limit": 1,
                "is_contact_sales": False,
                "is_featured": True,
                "sort_order": 1,
                "features": {
                    "scans_per_month": "500",
                    "diseases": "45-disease detection",
                    "explainability": "All methods (Grad-CAM, LIME, SHAP, Integrated Gradients, ELI5)",
                    "clinical_reasoning": True,
                    "knowledge_graph": True,
                    "voice_mode": True,
                    "report_retention_days": 365,
                    "watermark": False,
                    "pdf_export": True,
                    "support": "Email support",
                    "audit_log": False,
                    "team_seats": False,
                    "dhis2_fhir": False,
                    "sso": False,
                },
            },
            {
                "id": uuid.uuid4(),
                "code": "practice",
                "display_name": "Practice",
                "tagline": "For multi-clinician clinics with shared review queues.",
                "description": "3,000 scans/month, 5 seats included, audit log, review queue collaboration.",
                "monthly_price_cents": 14900,
                "annual_price_cents": 149000,  # $1,490/yr (17% off $1,788)
                "currency": "USD",
                "scan_limit_monthly": 3000,
                "seat_limit": 5,
                "is_contact_sales": False,
                "is_featured": False,
                "sort_order": 2,
                "features": {
                    "scans_per_month": "3,000",
                    "diseases": "45-disease detection",
                    "explainability": "All methods + comprehensive reports",
                    "clinical_reasoning": True,
                    "knowledge_graph": True,
                    "voice_mode": True,
                    "review_queue": True,
                    "report_retention_days": 1095,
                    "watermark": False,
                    "pdf_export": True,
                    "support": "Priority email + chat",
                    "audit_log": True,
                    "team_seats": 5,
                    "fairness_dashboard": True,
                    "dhis2_fhir": False,
                    "sso": False,
                },
            },
            {
                "id": uuid.uuid4(),
                "code": "health_system",
                "display_name": "Health System",
                "tagline": "For hospitals, regional health offices, and NGOs.",
                "description": "Unlimited scans, SSO/SCIM, BAA/PDP, DHIS2 + FHIR + DICOM integrations, dedicated CSM.",
                "monthly_price_cents": None,
                "annual_price_cents": None,
                "currency": "USD",
                "scan_limit_monthly": None,
                "seat_limit": None,
                "is_contact_sales": True,
                "is_featured": False,
                "sort_order": 3,
                "features": {
                    "scans_per_month": "Unlimited",
                    "diseases": "45-disease detection",
                    "explainability": "All methods + governance reports",
                    "clinical_reasoning": True,
                    "knowledge_graph": True,
                    "voice_mode": True,
                    "review_queue": True,
                    "report_retention_days": "Custom",
                    "watermark": False,
                    "pdf_export": True,
                    "support": "Dedicated CSM + SLA",
                    "audit_log": True,
                    "team_seats": "Unlimited",
                    "fairness_dashboard": True,
                    "dhis2_fhir": True,
                    "dicom": True,
                    "sms_referral": True,
                    "sso": True,
                    "scim": True,
                    "baa": True,
                    "pdp_act": True,
                },
            },
        ],
    )


def downgrade() -> None:
    op.drop_table("organization_invites")
    op.drop_table("magic_link_tokens")
    op.drop_table("password_reset_tokens")
    op.drop_table("email_verification_tokens")
    op.drop_table("refresh_tokens")
    op.drop_table("webhook_events")
    op.drop_table("usage_events")
    op.drop_table("payment_intents")
    op.drop_table("invoices")
    op.drop_table("subscriptions")
    op.drop_table("memberships")
    op.drop_table("organizations")
    op.drop_table("plans")
    op.drop_table("users")

    for enum_name in [
        "usage_event_type",
        "payment_intent_status",
        "invoice_status",
        "payment_provider",
        "billing_cycle",
        "subscription_status",
        "membership_status",
        "membership_role",
    ]:
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)
