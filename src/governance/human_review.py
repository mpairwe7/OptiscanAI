"""Human-in-the-loop review system for low-confidence and high-risk predictions."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ReviewReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_SEVERITY = "high_severity"
    CONFLICTING_PREDICTIONS = "conflicting_predictions"
    DRIFT_DETECTED = "drift_detected"
    RARE_DISEASE = "rare_disease"
    REFERRAL_URGENT = "referral_urgent"


class ReviewDecision(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    MODIFIED = "modified"
    ESCALATED = "escalated"


@dataclass
class ReviewRequest:
    request_id: str
    prediction_id: str
    reason: str
    priority: str  # low | medium | high | urgent
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prediction_summary: dict = field(default_factory=dict)
    reviewer: Optional[str] = None
    decision: Optional[str] = None
    notes: str = ""
    resolved_at: Optional[str] = None


class HumanReviewGate:
    """Determines whether a prediction needs human review."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_confidence_for_review: float = 0.7,
        urgent_referral_diseases: list[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_confidence = max_confidence_for_review
        self.urgent_diseases = urgent_referral_diseases or [
            "DR",
            "ARMD",
            "CRVO",
            "CRAO",
            "AION",
            "VH",
            "RS",  # sight-threatening
        ]
        self._pending_reviews: list[ReviewRequest] = []

    def check(self, prediction_result: dict) -> Optional[ReviewRequest]:
        """Evaluate whether a prediction needs human review.
        Returns a ReviewRequest if review needed, None otherwise."""
        reasons = []
        priority = "low"

        predictions = prediction_result.get("predictions", [])

        # Check 1: Low overall confidence
        if predictions:
            max_prob = max(p["probability"] for p in predictions)
            if max_prob < self.max_confidence:
                reasons.append(ReviewReason.LOW_CONFIDENCE)
                priority = "medium"

        # Check 2: Borderline predictions (close to threshold)
        threshold = prediction_result.get("threshold", 0.5)
        per_class_thresholds = prediction_result.get("per_class_thresholds", {})
        borderline = [
            p
            for p in predictions
            if abs(p["probability"] - per_class_thresholds.get(p["code"], threshold)) < 0.1
        ]
        if len(borderline) > 2:
            reasons.append(ReviewReason.CONFLICTING_PREDICTIONS)
            priority = "medium"

        # Check 3: Urgent referral diseases detected
        detected_codes = [p["code"] for p in predictions]
        urgent_detected = [c for c in detected_codes if c in self.urgent_diseases]
        if urgent_detected:
            reasons.append(ReviewReason.HIGH_SEVERITY)
            priority = "high"
            if prediction_result.get("clinical", {}).get("referral_priority") == "URGENT":
                reasons.append(ReviewReason.REFERRAL_URGENT)
                priority = "urgent"

        # Check 4: No diseases detected but model is uncertain
        if not predictions and prediction_result.get("model_loaded", False):
            all_probs = prediction_result.get("all_probabilities", {})
            near_threshold = sum(
                1
                for code, value in all_probs.items()
                if isinstance(value, dict)
                and abs(
                    value.get("probability", 0)
                    - value.get("threshold", per_class_thresholds.get(code, threshold))
                )
                < 0.15
            )
            if near_threshold > 5:
                reasons.append(ReviewReason.LOW_CONFIDENCE)
                priority = "medium"

        if not reasons:
            return None

        request = ReviewRequest(
            request_id=f"review_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            prediction_id=prediction_result.get("request_id", "unknown"),
            reason=",".join(r.value for r in reasons),
            priority=priority,
            prediction_summary={
                "num_detected": prediction_result.get("total_detected", 0),
                "top_predictions": [
                    {"code": p["code"], "probability": p["probability"]} for p in predictions[:5]
                ],
                "referral": prediction_result.get("clinical", {}).get("referral_priority", ""),
            },
        )

        self._pending_reviews.append(request)
        logger.info(
            f"Human review requested: {request.request_id} priority={priority} reasons={request.reason}"
        )
        return request

    @property
    def pending_count(self) -> int:
        return len([r for r in self._pending_reviews if r.decision is None])

    def resolve(
        self, request_id: str, reviewer: str, decision: ReviewDecision, notes: str = ""
    ) -> bool:
        """Mark a review request as resolved."""
        for req in self._pending_reviews:
            if req.request_id == request_id:
                req.reviewer = reviewer
                req.decision = decision.value
                req.notes = notes
                req.resolved_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"Review {request_id} resolved by {reviewer}: {decision.value}")
                return True
        return False

    def get_pending(self, priority: str = None) -> list[ReviewRequest]:
        """Get pending review requests, optionally filtered by priority."""
        pending = [r for r in self._pending_reviews if r.decision is None]
        if priority:
            pending = [r for r in pending if r.priority == priority]
        return sorted(
            pending,
            key=lambda r: {"urgent": 0, "high": 1, "medium": 2, "low": 3}.get(r.priority, 4),
        )
