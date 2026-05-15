"""Human-in-the-loop review API endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.governance.human_review import HumanReviewGate, ReviewDecision

from fastapi import Depends
from backend.app.core.feature_gate import require_tier

router = APIRouter(
    prefix="/api/v1/review",
    tags=["human-review"],
    dependencies=[Depends(require_tier("practice", feature="review_queue"))],
)

# Global review gate (shared with prediction pipeline)
review_gate = HumanReviewGate()


class ReviewDecisionRequest(BaseModel):
    reviewer: str
    decision: str  # confirmed | rejected | modified | escalated
    notes: str = ""
    corrected_labels: dict[str, float] | None = None  # Phase 1: corrected disease labels
    image_path: str = ""  # Phase 1: path to the original image


@router.get("/pending")
async def get_pending_reviews(priority: Optional[str] = None):
    """Get pending review requests."""
    pending = review_gate.get_pending(priority=priority)
    return {
        "total_pending": review_gate.pending_count,
        "reviews": [
            {
                "request_id": r.request_id,
                "prediction_id": r.prediction_id,
                "reason": r.reason,
                "priority": r.priority,
                "created_at": r.created_at,
                "summary": r.prediction_summary,
            }
            for r in pending
        ],
    }


@router.post("/{request_id}/resolve")
async def resolve_review(request_id: str, body: ReviewDecisionRequest):
    """Resolve a pending review."""
    try:
        decision = ReviewDecision(body.decision)
    except ValueError:
        raise HTTPException(400, f"Invalid decision. Must be one of: {[d.value for d in ReviewDecision]}")

    success = review_gate.resolve(request_id, body.reviewer, decision, body.notes)
    if not success:
        raise HTTPException(404, f"Review {request_id} not found")

    # Phase 1: Active learning hook — queue corrected samples for fine-tuning
    al_result = None
    if success and body.decision == "modified":
        try:
            from backend.app.core.active_learning import get_active_learning_loop

            loop = get_active_learning_loop()
            if loop is not None:
                al_result = await loop.on_review_resolved(
                    request_id=request_id,
                    decision=body.decision,
                    reviewer=body.reviewer,
                    corrected_labels=body.corrected_labels,
                    image_path=body.image_path,
                    notes=body.notes,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                f"Active learning hook failed (non-fatal): {e}", exc_info=True
            )

    # Phase 1: Record review metric in OpenTelemetry
    try:
        from backend.app.core.telemetry import record_review_metric
        record_review_metric(body.decision)
    except Exception:
        pass

    result = {"status": "resolved", "request_id": request_id, "decision": body.decision}
    if al_result:
        result["active_learning"] = al_result
    return result


@router.get("/stats")
async def review_stats():
    """Get review statistics."""
    all_reviews = review_gate._pending_reviews
    return {
        "total": len(all_reviews),
        "pending": review_gate.pending_count,
        "resolved": len([r for r in all_reviews if r.decision is not None]),
        "by_priority": {
            p: len([r for r in all_reviews if r.priority == p and r.decision is None])
            for p in ["urgent", "high", "medium", "low"]
        },
    }
