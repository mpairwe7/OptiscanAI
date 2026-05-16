"""Billing routes: public plans catalog + authenticated subscription/usage ops."""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import AuthContext, get_auth_context, require_superuser
from backend.app.core.db import get_db
from backend.app.models.membership import Membership, MembershipRole, MembershipStatus
from backend.app.models.plan import Plan
from backend.app.models.subscription import BillingCycle
from backend.app.models.usage_event import UsageEventType
from backend.app.schemas.billing import (
    ChangePlanRequest,
    InvoiceResponse,
    PlanResponse,
    SubscriptionResponse,
    UsageResponse,
)
from backend.app.services.billing_service import (
    cancel_at_period_end,
    change_plan_immediately,
    get_active_subscription,
    list_invoices,
    resume_subscription,
    roll_period_forward_if_needed,
)
from backend.app.services.usage_service import (
    breakdown_by_event_type,
    count_events_in_period,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/billing", tags=["billing"])


# ── Public ──

@router.get("/plans", response_model=list[PlanResponse])
async def list_plans(db: AsyncSession = Depends(get_db)) -> list[PlanResponse]:
    stmt = select(Plan).where(Plan.is_active.is_(True)).order_by(Plan.sort_order)
    plans = (await db.execute(stmt)).scalars().all()
    return [
        PlanResponse(
            code=p.code, display_name=p.display_name,
            description=p.description, tagline=p.tagline,
            monthly_price_cents=p.monthly_price_cents,
            annual_price_cents=p.annual_price_cents,
            currency=p.currency,
            scan_limit_monthly=p.scan_limit_monthly,
            seat_limit=p.seat_limit,
            is_contact_sales=p.is_contact_sales,
            is_featured=p.is_featured,
            features=p.features or {},
        )
        for p in plans
    ]


# ── Subscription ──

@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    sub = await get_active_subscription(db, str(ctx.organization.id))
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription")
    await roll_period_forward_if_needed(db, sub)
    plan = await db.get(Plan, sub.plan_id)
    return SubscriptionResponse(
        plan_code=plan.code,
        plan_display_name=plan.display_name,
        status=sub.status.value,
        billing_cycle=sub.billing_cycle.value,
        provider=sub.provider.value,
        current_period_start=sub.current_period_start,
        current_period_end=sub.current_period_end,
        cancel_at_period_end=sub.cancel_at_period_end,
        canceled_at=sub.canceled_at,
    )


@router.post("/subscription/change", response_model=SubscriptionResponse)
async def change_subscription(
    body: ChangePlanRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only org owners/admins can change the plan")

    if body.plan_code == "health_system":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "contact_sales_required",
                "message": "Health System tier is sold via contact sales.",
                "contact_url": "/contact-sales",
            },
        )

    cycle = BillingCycle(body.billing_cycle)

    # Phase C — only Free → Free or paid → Free is fully self-serve here.
    # Paid upgrades route through /api/v1/payments/* in Phase D/E.
    if body.plan_code != "free":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "payment_required",
                "message": "Paid plan changes require checkout.",
                "checkout_url": f"/app/checkout/{body.plan_code}?cycle={cycle.value}",
            },
        )

    sub = await change_plan_immediately(
        db, organization=ctx.organization, plan_code=body.plan_code, cycle=cycle,
    )
    plan = await db.get(Plan, sub.plan_id)
    return SubscriptionResponse(
        plan_code=plan.code, plan_display_name=plan.display_name,
        status=sub.status.value, billing_cycle=sub.billing_cycle.value,
        provider=sub.provider.value,
        current_period_start=sub.current_period_start,
        current_period_end=sub.current_period_end,
        cancel_at_period_end=sub.cancel_at_period_end,
        canceled_at=sub.canceled_at,
    )


@router.post("/subscription/cancel", response_model=SubscriptionResponse)
async def cancel_subscription(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can cancel")
    sub = await cancel_at_period_end(db, ctx.organization)
    plan = await db.get(Plan, sub.plan_id)
    return SubscriptionResponse(
        plan_code=plan.code, plan_display_name=plan.display_name,
        status=sub.status.value, billing_cycle=sub.billing_cycle.value,
        provider=sub.provider.value,
        current_period_start=sub.current_period_start,
        current_period_end=sub.current_period_end,
        cancel_at_period_end=sub.cancel_at_period_end,
        canceled_at=sub.canceled_at,
    )


@router.post("/subscription/resume", response_model=SubscriptionResponse)
async def resume(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SubscriptionResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can resume")
    sub = await resume_subscription(db, ctx.organization)
    plan = await db.get(Plan, sub.plan_id)
    return SubscriptionResponse(
        plan_code=plan.code, plan_display_name=plan.display_name,
        status=sub.status.value, billing_cycle=sub.billing_cycle.value,
        provider=sub.provider.value,
        current_period_start=sub.current_period_start,
        current_period_end=sub.current_period_end,
        cancel_at_period_end=sub.cancel_at_period_end,
        canceled_at=sub.canceled_at,
    )


# ── Usage ──

@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> UsageResponse:
    sub = await get_active_subscription(db, str(ctx.organization.id))
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription")
    await roll_period_forward_if_needed(db, sub)
    plan = await db.get(Plan, sub.plan_id)

    used = await count_events_in_period(
        db,
        organization_id=str(ctx.organization.id),
        event_type=UsageEventType.SCAN,
        period_start=sub.current_period_start,
        period_end=sub.current_period_end,
    )
    breakdown = await breakdown_by_event_type(
        db,
        organization_id=str(ctx.organization.id),
        period_start=sub.current_period_start,
        period_end=sub.current_period_end,
    )

    seats_used = int(
        (await db.execute(
            select(func.count())
            .select_from(Membership)
            .where(
                Membership.organization_id == ctx.organization.id,
                Membership.status == MembershipStatus.ACTIVE,
            )
        )).scalar_one()
    )

    return UsageResponse(
        period_start=sub.current_period_start,
        period_end=sub.current_period_end,
        scan_limit=plan.scan_limit_monthly,
        scans_used=used,
        scans_remaining=None if plan.scan_limit_monthly is None else max(0, plan.scan_limit_monthly - used),
        seat_limit=plan.seat_limit,
        seats_used=seats_used,
        breakdown=breakdown,
    )


# ── Invoices ──

@router.get("/invoices", response_model=list[InvoiceResponse])
async def get_invoices(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[InvoiceResponse]:
    rows = await list_invoices(db, str(ctx.organization.id))
    return [
        InvoiceResponse(
            id=str(i.id),
            amount_cents=i.amount_cents,
            currency=i.currency,
            status=i.status.value,
            provider=i.provider.value,
            hosted_url=i.hosted_url,
            pdf_url=i.pdf_url,
            period_start=i.period_start,
            period_end=i.period_end,
            issued_at=i.issued_at,
            paid_at=i.paid_at,
        )
        for i in rows
    ]


# ── Practice seats ──


class SeatStateResponse(BaseModel):
    included_seats: int
    additional_seats: int
    effective_limit: int | None
    seats_used: int
    can_buy_more: bool
    per_seat_cents: int
    cycle: str


class UpdateSeatsRequest(BaseModel):
    additional_seats: int = Field(ge=0, le=500)


def _seat_response(state) -> "SeatStateResponse":
    return SeatStateResponse(
        included_seats=state.included_seats,
        additional_seats=state.additional_seats,
        effective_limit=state.effective_limit,
        seats_used=state.seats_used,
        can_buy_more=state.can_buy_more,
        per_seat_cents=state.per_seat_cents,
        cycle=state.cycle,
    )


@router.get("/seats", response_model=SeatStateResponse)
async def get_seats(
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SeatStateResponse:
    from backend.app.services.seat_service import get_seat_state
    sub = await get_active_subscription(db, str(ctx.organization.id))
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription")
    state = await get_seat_state(db, subscription=sub)
    return _seat_response(state)


@router.post("/seats", response_model=SeatStateResponse)
async def update_seats(
    body: UpdateSeatsRequest,
    ctx: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SeatStateResponse:
    if ctx.role not in {MembershipRole.OWNER.value, MembershipRole.ADMIN.value}:
        raise HTTPException(status_code=403, detail="Only owners/admins can change seat count")
    from backend.app.services.seat_service import get_seat_state, update_seat_quantity
    sub = await update_seat_quantity(
        db,
        organization=ctx.organization,
        target_total_additional_seats=body.additional_seats,
    )
    state = await get_seat_state(db, subscription=sub)
    return _seat_response(state)


# ── Admin: webhook replay ops view ──


class WebhookEventDTO(BaseModel):
    id: str
    provider: str
    provider_event_id: str
    event_type: str | None
    has_payload: bool
    received_at: datetime
    processed_at: datetime | None
    error: str | None


@router.get("/admin/webhook-events", include_in_schema=False)
async def list_webhook_events(
    provider: str | None = None,
    state: str | None = None,
    limit: int = 100,
    _ctx: AuthContext = Depends(require_superuser),
    db: AsyncSession = Depends(get_db),
) -> list[WebhookEventDTO]:
    """Recent webhook events for the ops UI."""
    from backend.app.models.subscription import PaymentProvider
    from backend.app.models.webhook_event import WebhookEvent

    stmt = select(WebhookEvent).order_by(WebhookEvent.received_at.desc()).limit(min(limit, 500))
    if provider:
        try:
            stmt = stmt.where(WebhookEvent.provider == PaymentProvider(provider))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown provider {provider!r}")
    if state == "error":
        stmt = stmt.where(WebhookEvent.error.is_not(None))
    elif state == "ok":
        stmt = stmt.where(WebhookEvent.error.is_(None), WebhookEvent.processed_at.is_not(None))
    elif state == "pending":
        stmt = stmt.where(WebhookEvent.processed_at.is_(None))

    rows = (await db.execute(stmt)).scalars().all()
    return [
        WebhookEventDTO(
            id=str(r.id),
            provider=r.provider.value,
            provider_event_id=r.provider_event_id,
            event_type=r.event_type,
            has_payload=bool(r.payload),
            received_at=r.received_at,
            processed_at=r.processed_at,
            error=r.error,
        )
        for r in rows
    ]


@router.get("/admin/webhook-events/{event_id}", include_in_schema=False)
async def get_webhook_event(
    event_id: str,
    _ctx: AuthContext = Depends(require_superuser),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Full payload for one event — for the JSON drawer."""
    from backend.app.models.webhook_event import WebhookEvent
    row = await db.get(WebhookEvent, event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return {
        "id": str(row.id),
        "provider": row.provider.value,
        "provider_event_id": row.provider_event_id,
        "event_type": row.event_type,
        "received_at": row.received_at.isoformat(),
        "processed_at": row.processed_at.isoformat() if row.processed_at else None,
        "error": row.error,
        "payload": row.payload,
    }


@router.post("/admin/webhook-events/{event_id}/replay", include_in_schema=False)
async def replay_webhook_event(
    event_id: str,
    _ctx: AuthContext = Depends(require_superuser),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from backend.app.services.webhook_replay_service import replay_webhook
    return await replay_webhook(db, event_id)


# ── Admin: manual renewal-reminder trigger ──

@router.post("/admin/run-renewal-reminders", include_in_schema=False)
async def admin_run_renewal_reminders(
    _ctx: AuthContext = Depends(require_superuser),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Run the MoMo/Flutterwave renewal-reminder cron synchronously.

    Idempotent — re-running before subscription period_end advances is a no-op.
    Restricted to platform superusers (User.is_superuser).
    """
    from backend.app.services.renewal_service import run_renewal_reminders
    result = await run_renewal_reminders(db)
    return {"status": "ok", **result.as_dict()}
