"""Pydantic request/response schemas for /api/v1/auth."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    full_name: Optional[str] = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class MagicLinkRequest(BaseModel):
    email: EmailStr


class PasswordForgotRequest(BaseModel):
    email: EmailStr


class PasswordResetRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=200)


class TokenPairResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class OrgSummary(BaseModel):
    id: str
    name: str
    slug: str
    is_personal: bool


class PlanSummary(BaseModel):
    code: str
    display_name: str
    scan_limit_monthly: Optional[int]
    seat_limit: Optional[int]


class SubscriptionSummary(BaseModel):
    plan: PlanSummary
    status: str
    billing_cycle: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool


class MeResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    email_verified: bool
    is_superuser: bool = False
    organization: OrgSummary
    role: str
    subscription: Optional[SubscriptionSummary]


class AuthSuccessResponse(BaseModel):
    user: MeResponse
    access_token: str
    token_type: str = "bearer"
    expires_in: int
