"""Pydantic schemas for /api/v1/billing."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PlanResponse(BaseModel):
    code: str
    display_name: str
    description: Optional[str]
    tagline: Optional[str]
    monthly_price_cents: Optional[int]
    annual_price_cents: Optional[int]
    currency: str
    scan_limit_monthly: Optional[int]
    seat_limit: Optional[int]
    is_contact_sales: bool
    is_featured: bool
    features: dict


class SubscriptionResponse(BaseModel):
    plan_code: str
    plan_display_name: str
    status: str
    billing_cycle: str
    provider: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    canceled_at: Optional[datetime]


class UsageResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    scan_limit: Optional[int]  # None = unlimited
    scans_used: int
    scans_remaining: Optional[int]
    seat_limit: Optional[int]
    seats_used: int
    breakdown: dict[str, int]  # event_type → count


class ChangePlanRequest(BaseModel):
    plan_code: str = Field(pattern="^(free|clinician|practice|health_system)$")
    billing_cycle: str = Field(default="monthly", pattern="^(monthly|annual)$")


class InvoiceResponse(BaseModel):
    id: str
    amount_cents: int
    currency: str
    status: str
    provider: str
    hosted_url: Optional[str]
    pdf_url: Optional[str]
    period_start: datetime
    period_end: datetime
    issued_at: datetime
    paid_at: Optional[datetime]
