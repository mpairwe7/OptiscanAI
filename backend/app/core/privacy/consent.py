"""Uganda PDP Act 2019 consent recording and management.

Records patient consent with timestamp, method, purpose, and scope.
Supports voice consent (audio recording + transcript with SHA-256 hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConsentRecord:
    """Recorded patient consent."""
    consent_id: str
    patient_id_hash: str  # SHA-256 hashed patient identifier
    purpose: str  # screening | referral | data_sharing | research
    scope: list[str] = field(default_factory=list)  # image | voice | demographics
    consent_method: str = "voice"  # voice | written | digital
    language: str = "en"
    granted: bool = True
    timestamp: float = 0.0
    audio_hash: str = ""  # SHA-256 of voice consent recording
    transcript: str = ""
    entry_hash: str = ""  # Integrity hash

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class ConsentManager:
    """Uganda PDP Act 2019 consent recording."""

    def __init__(self, storage_dir: str = "data/consent"):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._consent_file = self._storage_dir / "consent_log.jsonl"

    async def record_consent(
        self,
        patient_id: str,
        purpose: str,
        scope: list[str],
        consent_method: str = "voice",
        language: str = "en",
        audio_bytes: Optional[bytes] = None,
        transcript: str = "",
    ) -> ConsentRecord:
        """Record explicit patient consent."""
        from uuid import uuid4

        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest() if audio_bytes else ""

        record = ConsentRecord(
            consent_id=str(uuid4()),
            patient_id_hash=patient_hash,
            purpose=purpose,
            scope=scope,
            consent_method=consent_method,
            language=language,
            audio_hash=audio_hash,
            transcript=transcript,
        )

        # Compute integrity hash
        hash_input = json.dumps({
            "consent_id": record.consent_id,
            "patient_id_hash": record.patient_id_hash,
            "purpose": record.purpose,
            "timestamp": record.timestamp,
        }, sort_keys=True)
        record.entry_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        # Persist
        with open(self._consent_file, "a") as f:
            f.write(json.dumps({
                "consent_id": record.consent_id,
                "patient_id_hash": record.patient_id_hash,
                "purpose": record.purpose,
                "scope": record.scope,
                "method": record.consent_method,
                "language": record.language,
                "granted": record.granted,
                "timestamp": record.timestamp,
                "audio_hash": record.audio_hash,
                "entry_hash": record.entry_hash,
            }) + "\n")

        # Save audio recording if provided
        if audio_bytes:
            audio_path = self._storage_dir / f"{record.consent_id}.wav"
            audio_path.write_bytes(audio_bytes)

        logger.info(
            "Consent recorded: %s (purpose=%s, method=%s)",
            record.consent_id, purpose, consent_method,
        )
        return record

    async def verify_consent(self, patient_id: str, purpose: str) -> bool:
        """Check if valid consent exists for the given purpose."""
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()

        if not self._consent_file.exists():
            return False

        with open(self._consent_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                if (
                    entry.get("patient_id_hash") == patient_hash
                    and entry.get("purpose") == purpose
                    and entry.get("granted", False)
                ):
                    return True
        return False

    async def revoke_consent(self, patient_id: str, purpose: str) -> None:
        """Revoke consent and log the revocation."""
        record = await self.record_consent(
            patient_id=patient_id,
            purpose=purpose,
            scope=[],
            consent_method="digital",
        )
        # Mark as revoked
        with open(self._consent_file, "a") as f:
            f.write(json.dumps({
                "consent_id": record.consent_id,
                "patient_id_hash": record.patient_id_hash,
                "purpose": purpose,
                "granted": False,
                "timestamp": time.time(),
                "action": "revoked",
            }) + "\n")
        logger.info("Consent revoked for purpose=%s", purpose)
