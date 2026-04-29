"""
ActiveLearningManager — Human-in-the-loop active learning for retinal AI.

Workflow:
    1. Model makes prediction with uncertainty estimate
    2. Low-confidence / high-uncertainty cases flagged for human review
    3. Ophthalmologist corrects prediction via review interface
    4. Corrected sample added to fine-tuning queue
    5. When queue reaches threshold (200 samples), trigger incremental LoRA fine-tuning
    6. New model version evaluated and promoted if improved

Sampling strategies:
    - Uncertainty sampling (highest epistemic uncertainty)
    - Margin sampling (smallest gap between top-2 predictions)
    - Diversity sampling (maximize feature-space coverage)
    - Clinical priority (sight-threatening conditions first)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class SamplingStrategy(str, Enum):
    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    DIVERSITY = "diversity"
    CLINICAL_PRIORITY = "clinical_priority"
    COMBINED = "combined"


@dataclass
class ReviewCase:
    """A single case flagged for human review."""
    case_id: str
    image_path: str
    model_predictions: Dict[str, float]
    uncertainty: Dict[str, float]
    model_version: str
    timestamp: float = field(default_factory=time.time)
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_id: Optional[str] = None
    corrected_labels: Optional[Dict[str, float]] = None
    reviewer_notes: Optional[str] = None
    review_timestamp: Optional[float] = None
    priority: str = "normal"
    sampling_reason: str = "uncertainty"


class ActiveLearningManager:
    """Manages the active learning loop between model and human reviewers.

    Parameters
    ----------
    queue_dir : str
        Directory to persist the review queue.
    confidence_threshold : float
        Predictions below this trigger review. Default 0.65.
    uncertainty_threshold : float
        Epistemic uncertainty above this triggers review. Default 0.3.
    retrain_threshold : int
        Number of corrected samples before triggering fine-tuning. Default 200.
    strategy : SamplingStrategy
        Active learning sampling strategy. Default COMBINED.
    max_queue_size : int
        Maximum pending review queue size. Default 5000.
    """

    def __init__(
        self,
        queue_dir: str = "data/active_learning",
        confidence_threshold: float = 0.65,
        uncertainty_threshold: float = 0.3,
        retrain_threshold: int = 200,
        strategy: SamplingStrategy = SamplingStrategy.COMBINED,
        max_queue_size: int = 5000,
    ):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.retrain_threshold = retrain_threshold
        self.strategy = strategy
        self.max_queue_size = max_queue_size

        # In-memory state
        self.pending_queue: List[ReviewCase] = []
        self.completed_reviews: List[ReviewCase] = []
        self.fine_tune_queue: List[ReviewCase] = []

        # Statistics
        self.stats = defaultdict(int)

        # Load persisted state
        self._load_state()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def should_flag_for_review(
        self,
        predictions: Dict[str, float],
        uncertainty: Dict[str, float],
    ) -> bool:
        """Determine if this prediction should be flagged for human review.

        A case is flagged if:
        - Max prediction confidence < confidence_threshold, OR
        - Mean epistemic uncertainty > uncertainty_threshold, OR
        - Clinical priority conditions detected with moderate confidence
        """
        max_conf = max(predictions.values()) if predictions else 0
        mean_uncertainty = np.mean(list(uncertainty.get("epistemic", {0: 0}).values()))

        # Low confidence
        if max_conf < self.confidence_threshold:
            return True

        # High uncertainty
        if mean_uncertainty > self.uncertainty_threshold:
            return True

        # Clinical priority: sight-threatening conditions at moderate confidence
        urgent_codes = {"DR", "CRVO", "CRAO", "VH", "ODC"}
        for code in urgent_codes:
            if code in predictions:
                conf = predictions[code]
                if 0.3 < conf < self.confidence_threshold:
                    return True

        return False

    def add_to_review_queue(
        self,
        case_id: str,
        image_path: str,
        predictions: Dict[str, float],
        uncertainty: Dict[str, float],
        model_version: str = "unknown",
    ) -> Optional[ReviewCase]:
        """Add a case to the review queue if it should be flagged.

        Returns the ReviewCase if added, None otherwise.
        """
        if not self.should_flag_for_review(predictions, uncertainty):
            return None

        if len(self.pending_queue) >= self.max_queue_size:
            logger.warning("Review queue full; dropping lowest-priority case")
            self._evict_lowest_priority()

        # Determine priority
        priority = self._compute_priority(predictions, uncertainty)
        reason = self._get_sampling_reason(predictions, uncertainty)

        case = ReviewCase(
            case_id=case_id,
            image_path=image_path,
            model_predictions=predictions,
            uncertainty=uncertainty,
            model_version=model_version,
            priority=priority,
            sampling_reason=reason,
        )

        self.pending_queue.append(case)
        self.stats["cases_flagged"] += 1
        self._persist_state()

        logger.info(f"Case {case_id} flagged for review (priority={priority}, reason={reason})")
        return case

    def submit_review(
        self,
        case_id: str,
        reviewer_id: str,
        corrected_labels: Optional[Dict[str, float]] = None,
        approved: bool = False,
        notes: Optional[str] = None,
    ) -> bool:
        """Submit a human review for a flagged case.

        Parameters
        ----------
        case_id : str
            ID of the case being reviewed.
        reviewer_id : str
            ID of the reviewing ophthalmologist.
        corrected_labels : dict | None
            Corrected disease labels if model was wrong.
        approved : bool
            True if model prediction was correct.
        notes : str | None
            Reviewer's clinical notes.

        Returns
        -------
        bool
            True if review was recorded successfully.
        """
        case = self._find_pending_case(case_id)
        if case is None:
            logger.warning(f"Case {case_id} not found in pending queue")
            return False

        case.reviewer_id = reviewer_id
        case.review_timestamp = time.time()
        case.reviewer_notes = notes

        if approved:
            case.status = ReviewStatus.APPROVED
            case.corrected_labels = case.model_predictions
        else:
            case.status = ReviewStatus.CORRECTED
            case.corrected_labels = corrected_labels or {}

        # Move to completed
        self.pending_queue.remove(case)
        self.completed_reviews.append(case)

        # Add to fine-tune queue if corrected
        if case.status == ReviewStatus.CORRECTED:
            self.fine_tune_queue.append(case)
            self.stats["corrections"] += 1

        self.stats["reviews_completed"] += 1
        self._persist_state()

        # Check if we should trigger retraining
        if len(self.fine_tune_queue) >= self.retrain_threshold:
            self._trigger_retraining()

        return True

    def get_next_review_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Get the next batch of cases for review, prioritized."""
        priority_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
        sorted_queue = sorted(
            self.pending_queue,
            key=lambda c: (priority_order.get(c.priority, 99), -c.timestamp),
        )

        batch = sorted_queue[:batch_size]
        for case in batch:
            case.status = ReviewStatus.IN_REVIEW

        return [asdict(c) for c in batch]

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def _compute_priority(self, predictions: Dict[str, float],
                          uncertainty: Dict[str, float]) -> str:
        """Compute review priority based on clinical severity and uncertainty."""
        # Urgent: sight-threatening conditions
        urgent_codes = {"DR", "CRVO", "CRAO", "VH"}
        for code in urgent_codes:
            if predictions.get(code, 0) > 0.3:
                return "urgent"

        # High: high uncertainty
        mean_unc = np.mean(list(uncertainty.get("epistemic", {0: 0}).values()))
        if mean_unc > self.uncertainty_threshold * 1.5:
            return "high"

        # Normal: standard low-confidence case
        max_conf = max(predictions.values()) if predictions else 0
        if max_conf < self.confidence_threshold * 0.7:
            return "high"

        return "normal"

    def _get_sampling_reason(self, predictions: Dict[str, float],
                             uncertainty: Dict[str, float]) -> str:
        max_conf = max(predictions.values()) if predictions else 0
        mean_unc = np.mean(list(uncertainty.get("epistemic", {0: 0}).values()))

        if max_conf < self.confidence_threshold:
            return "low_confidence"
        if mean_unc > self.uncertainty_threshold:
            return "high_uncertainty"
        return "clinical_priority"

    # ------------------------------------------------------------------
    # Retraining trigger
    # ------------------------------------------------------------------

    def _trigger_retraining(self):
        """Trigger incremental LoRA fine-tuning with corrected samples."""
        logger.info(
            f"Retraining triggered: {len(self.fine_tune_queue)} corrected samples available"
        )

        # Export fine-tuning data
        export_path = self.queue_dir / "finetune_data"
        export_path.mkdir(exist_ok=True)

        manifest = []
        for case in self.fine_tune_queue:
            manifest.append({
                "image_path": case.image_path,
                "labels": case.corrected_labels,
                "reviewer_id": case.reviewer_id,
                "original_predictions": case.model_predictions,
            })

        manifest_path = export_path / f"manifest_{int(time.time())}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.stats["retraining_triggers"] += 1
        self.fine_tune_queue.clear()
        self._persist_state()

        logger.info(f"Fine-tuning manifest saved to {manifest_path}")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _persist_state(self):
        """Save queue state to disk."""
        state = {
            "pending": [asdict(c) for c in self.pending_queue],
            "completed_count": len(self.completed_reviews),
            "finetune_queue_size": len(self.fine_tune_queue),
            "stats": dict(self.stats),
        }
        state_path = self.queue_dir / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """Load persisted state, including pending review queue."""
        state_path = self.queue_dir / "state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.stats.update(state.get("stats", {}))
                # Restore pending queue
                for item in state.get("pending", []):
                    try:
                        item["status"] = ReviewStatus(item.get("status", "pending"))
                        self.pending_queue.append(ReviewCase(**item))
                    except (TypeError, ValueError):
                        continue  # skip malformed entries
                logger.info(
                    f"Loaded active learning state: "
                    f"{len(self.pending_queue)} pending restored, "
                    f"{state.get('completed_count', 0)} completed"
                )
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _find_pending_case(self, case_id: str) -> Optional[ReviewCase]:
        for case in self.pending_queue:
            if case.case_id == case_id:
                return case
        return None

    def _evict_lowest_priority(self):
        """Remove lowest-priority (highest number) case from queue."""
        if self.pending_queue:
            priority_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
            # Sort ascending: urgent(0) first, low(3) last. pop() removes last = lowest priority.
            self.pending_queue.sort(
                key=lambda c: (priority_order.get(c.priority, 99), c.timestamp)
            )
            evicted = self.pending_queue.pop()
            self.stats["cases_evicted"] += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return active learning statistics."""
        return {
            "pending_reviews": len(self.pending_queue),
            "completed_reviews": len(self.completed_reviews),
            "fine_tune_queue": len(self.fine_tune_queue),
            "retrain_threshold": self.retrain_threshold,
            "progress_to_retrain": f"{len(self.fine_tune_queue)}/{self.retrain_threshold}",
            **dict(self.stats),
        }
