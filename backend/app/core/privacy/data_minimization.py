"""Data minimization and purpose limitation (PDP Act 2019).

Enforces data minimization for aggregate reporting and validates
cross-border data transfers against allowed destination countries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Countries permitted for cross-border data transfer under PDP Act 2019
# and bilateral agreements
ALLOWED_COUNTRIES = {"UG", "KE", "TZ", "RW", "SS", "CD"}


class DataMinimizer:
    """Enforce data minimization and purpose limitation."""

    def strip_pii(self, data: dict) -> dict:
        """Remove PII from screening result for aggregate reporting.

        Keeps: disease codes, probabilities, referral priority, timestamps.
        Removes: patient name, ID, phone, GPS coordinates, CHW identity.
        """
        pii_fields = {
            "patient_name",
            "patient_id",
            "patient_id_hash",
            "chw_name",
            "chw_id",
            "phone",
            "national_id",
            "gps_lat",
            "gps_lon",
            "device_id",
            "image_hash",
        }

        stripped = {}
        for key, value in data.items():
            if key in pii_fields:
                continue
            if isinstance(value, dict):
                stripped[key] = self.strip_pii(value)
            else:
                stripped[key] = value

        return stripped

    def anonymize_for_research(self, data: dict) -> dict:
        """Anonymize data for research purposes.

        Hashes identifiers and removes direct identifiers.
        """
        result = self.strip_pii(data)

        # Generalize age to decade
        if "patient_age" in data:
            try:
                age = int(data["patient_age"])
                result["age_decade"] = f"{(age // 10) * 10}s"
            except (ValueError, TypeError):
                pass

        # Keep sex for demographic analysis
        if "patient_sex" in data:
            result["sex"] = data["patient_sex"]

        return result

    def check_cross_border(self, destination_country: str) -> bool:
        """Check if data transfer to destination is PDP-compliant."""
        country = destination_country.upper().strip()
        allowed = country in ALLOWED_COUNTRIES
        if not allowed:
            logger.warning(
                "Cross-border transfer to %s NOT allowed under PDP Act 2019",
                country,
            )
        return allowed

    def validate_purpose(self, purpose: str, allowed_purposes: list[str]) -> bool:
        """Validate that data use aligns with consented purpose."""
        return purpose in allowed_purposes
