"""Pydantic models for DHIS2 API data types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrackedEntity(BaseModel):
    """DHIS2 Tracked Entity Instance (patient)."""

    tei_id: str = ""
    org_unit: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        first = self.attributes.get("first_name", "")
        last = self.attributes.get("last_name", "")
        return f"{first} {last}".strip()


class PatientRegistration(BaseModel):
    """Data for registering a new patient in DHIS2."""

    org_unit: str
    first_name: str
    last_name: str
    national_id: str = ""
    phone: str = ""
    sex: str = ""
    date_of_birth: str = ""


class ReferralEvent(BaseModel):
    """Screening referral event for DHIS2."""

    program: str = "RETINAL_SCREENING"
    org_unit: str
    tei_id: str
    event_date: str
    status: str = "ACTIVE"
    data_values: dict[str, str] = Field(default_factory=dict)


class AggregateReport(BaseModel):
    """Monthly aggregate screening report for DHIS2."""

    data_set: str
    period: str  # e.g., "202605"
    org_unit: str
    data_values: list[dict[str, str]] = Field(default_factory=list)
