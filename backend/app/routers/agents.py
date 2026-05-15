"""Agent orchestration API — status, control, event history, and agentic screening."""
import io
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

from fastapi import Depends
from backend.app.core.feature_gate import require_tier

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["agents"],
    dependencies=[Depends(require_tier("clinician", feature="agents"))],
)

# Global reference — set during app startup
_orchestrator = None


def set_orchestrator(orchestrator):
    global _orchestrator
    _orchestrator = orchestrator


def _get_orchestrator():
    if _orchestrator is None:
        raise HTTPException(503, "Agent orchestrator not initialized")
    return _orchestrator


@router.get("/status")
async def agent_status():
    """Get status of all autonomous agents."""
    orch = _get_orchestrator()
    return orch.status()


@router.get("/events")
async def agent_events(
    event_type: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
):
    """Query agent event history."""
    orch = _get_orchestrator()
    from src.agents.event_bus import EventType

    et = None
    if event_type:
        try:
            et = EventType(event_type)
        except ValueError:
            raise HTTPException(400, f"Invalid event type: {event_type}. Valid: {[e.value for e in EventType]}")

    events = orch.bus.get_history(event_type=et, source=source, limit=limit)
    return {
        "total": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "type": e.type.value,
                "source": e.source,
                "timestamp": e.timestamp,
                "data": e.data,
            }
            for e in events
        ],
    }


@router.get("/compliance")
async def compliance_report():
    """Generate on-demand compliance report from the governance agent."""
    orch = _get_orchestrator()
    gov = orch.governance
    if not gov:
        raise HTTPException(503, "Governance agent not available")

    result = await gov.use_tool("generate_compliance_report")
    if not result.success:
        raise HTTPException(500, result.error)
    return result.data


class ScreeningRequest(BaseModel):
    scan_id: str = ""


@router.get("/tools")
async def list_agent_tools():
    """List all tools available across all agents."""
    orch = _get_orchestrator()
    tools = {}
    for name, agent in orch._agents.items():
        tools[name] = agent.tools_available
    return {"agents": tools, "total_tools": sum(len(t) for t in tools.values())}


@router.post("/screen")
async def agentic_screen(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """Run the full LangGraph + Claude agentic screening pipeline.

    This is the agentic alternative to /api/v1/predict. Instead of just
    running inference, it orchestrates a multi-step workflow:
    classify → triage (Claude) → reason (KG) → explain (conditional) → review (conditional) → report (Claude)

    Returns a complete clinical screening report with:
    - Disease predictions + clinical reasoning
    - Claude-generated clinical narrative (when API available)
    - Triage decisions with reasoning
    - Explainability artifacts (if triggered)
    - Human review status (if flagged)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG)")

    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size // 1024 // 1024}MB)")

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # Fundus image gating (v2 fusion gate)
    from src.data.fundus_gate_v2 import gate_image
    gate_result = gate_image(image)
    if not gate_result.passed:
        detail = {
            "error": "non_fundus_image",
            "message": gate_result.reason,
            "confidence": gate_result.confidence,
            "layer": gate_result.layer,
            "checks": gate_result.checks,
        }
        if hasattr(gate_result, "failed_checks") and gate_result.failed_checks:
            detail["failed_checks"] = gate_result.failed_checks
        if hasattr(gate_result, "suggested_action") and gate_result.suggested_action:
            detail["suggestion"] = gate_result.suggested_action
        raise HTTPException(422, detail=detail)

    from src.agents.graph import run_screening
    import uuid

    scan_id = str(uuid.uuid4())[:8]
    try:
        report = await run_screening(image, scan_id=scan_id, threshold=threshold)
    except Exception as e:
        logger.error(f"Agentic screening failed: {e}", exc_info=True)
        raise HTTPException(500, f"Screening pipeline failed: {type(e).__name__}")

    if "error" in report and report.get("status") == "error":
        raise HTTPException(503, report["error"])

    return {"success": True, "scan_id": scan_id, "report": report}


@router.get("/graph/info")
async def graph_info():
    """Describe the LangGraph screening workflow topology."""
    from src.agents import llm

    return {
        "framework": "LangGraph + Multi-LLM",
        "graph_nodes": ["classify", "triage", "reason", "explain", "review", "report"],
        "conditional_edges": {
            "reason → explain|review|report": "3-way branch based on triage decisions",
            "explain → review|report": "2-way branch after explainability",
        },
        "llm_nodes": ["triage", "report"],
        "deterministic_nodes": ["classify", "reason", "explain", "review"],
        "llm_available": llm.is_available(),
        "active_provider": llm.get_provider(),
        "active_model": llm.get_model(),
        "fallback_chain": ["claude", "groq", "deterministic_rules"],
    }
