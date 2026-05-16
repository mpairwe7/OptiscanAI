"""Pydantic schemas for /api/v1/orgs."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class CreateOrgRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="clinician", pattern="^(admin|clinician|viewer)$")


class AcceptInviteRequest(BaseModel):
    token: str
    full_name: Optional[str] = Field(default=None, max_length=200)
    password: Optional[str] = Field(default=None, min_length=8, max_length=200)


class UpdateMemberRoleRequest(BaseModel):
    role: str = Field(pattern="^(admin|clinician|viewer)$")


class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    is_personal: bool
    is_active: bool
    role: str  # caller's role in this org
    created_at: datetime


class MemberResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str]
    role: str
    status: str
    joined_at: Optional[datetime]


class InviteResponse(BaseModel):
    id: str
    email: str
    role: str
    expires_at: datetime
    created_at: datetime
