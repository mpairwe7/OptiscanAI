"""DHIS2 API client for Uganda health information system.

Wraps the DHIS2 Web API for patient lookup, referral creation,
enrollment, and aggregate reporting.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import aiohttp

from .auth import DHIS2Auth
from .models import AggregateReport, PatientRegistration, ReferralEvent, TrackedEntity

logger = logging.getLogger(__name__)


class DHIS2Client:
    """Async HTTP client for Uganda DHIS2 instance."""

    def __init__(self, base_url: str, auth: DHIS2Auth, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = await self.auth.get_headers()
            headers["Content-Type"] = "application/json"
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=self.timeout
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- Patient lookup (TrackedEntityInstance API) --

    async def search_patient(
        self, query: str, org_unit: str
    ) -> list[TrackedEntity]:
        """Search for patients by name or NIN."""
        session = await self._get_session()
        url = f"{self.base_url}/api/trackedEntityInstances"
        params = {"ou": org_unit, "query": query, "pageSize": 20}

        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.error("DHIS2 search failed: %s", await resp.text())
                return []
            data = await resp.json()
            instances = data.get("trackedEntityInstances", [])
            return [
                TrackedEntity(
                    tei_id=i.get("trackedEntityInstance", ""),
                    org_unit=i.get("orgUnit", ""),
                    attributes={
                        a["displayName"]: a.get("value", "")
                        for a in i.get("attributes", [])
                    },
                )
                for i in instances
            ]

    async def get_patient(self, tei_id: str) -> Optional[TrackedEntity]:
        """Get a single patient by TEI ID."""
        session = await self._get_session()
        url = f"{self.base_url}/api/trackedEntityInstances/{tei_id}"

        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return TrackedEntity(
                tei_id=data.get("trackedEntityInstance", ""),
                org_unit=data.get("orgUnit", ""),
                attributes={
                    a["displayName"]: a.get("value", "")
                    for a in data.get("attributes", [])
                },
            )

    async def create_patient(self, patient: PatientRegistration) -> str:
        """Register a new patient. Returns TEI ID."""
        session = await self._get_session()
        url = f"{self.base_url}/api/trackedEntityInstances"

        payload = {
            "orgUnit": patient.org_unit,
            "trackedEntityType": "nEenWmSyUEp",  # Person
            "attributes": [
                {"attribute": "first_name", "value": patient.first_name},
                {"attribute": "last_name", "value": patient.last_name},
            ],
        }
        if patient.phone:
            payload["attributes"].append({"attribute": "phone", "value": patient.phone})

        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            tei_id = data.get("response", {}).get("importSummaries", [{}])[0].get("reference", "")
            logger.info("Created patient TEI: %s", tei_id)
            return tei_id

    # -- Referral creation (Events API) --

    async def create_referral_event(self, referral: ReferralEvent) -> str:
        """Create a referral event from screening result."""
        session = await self._get_session()
        url = f"{self.base_url}/api/events"

        payload = {
            "program": referral.program,
            "orgUnit": referral.org_unit,
            "trackedEntityInstance": referral.tei_id,
            "eventDate": referral.event_date,
            "status": referral.status,
            "dataValues": [
                {"dataElement": k, "value": v}
                for k, v in referral.data_values.items()
            ],
        }

        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            event_id = data.get("response", {}).get("importSummaries", [{}])[0].get("reference", "")
            logger.info("Created referral event: %s", event_id)
            return event_id

    # -- Aggregate reporting (DataValueSets API) --

    async def submit_aggregate_report(self, report: AggregateReport) -> dict:
        """Submit monthly aggregate screening report."""
        session = await self._get_session()
        url = f"{self.base_url}/api/dataValueSets"

        payload = {
            "dataSet": report.data_set,
            "period": report.period,
            "orgUnit": report.org_unit,
            "dataValues": report.data_values,
        }

        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            logger.info("Submitted aggregate report: period=%s", report.period)
            return data

    # -- Health check --

    async def ping(self) -> bool:
        """Check DHIS2 server connectivity."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/system/info") as resp:
                return resp.status == 200
        except Exception:
            return False
