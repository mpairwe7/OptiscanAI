"""LangGraph + Claude integration for RetinalAI clinical screening.

Architecture:
    LangGraph StateGraph manages the workflow (state, branching, cycles)
    Claude provides reasoning at decision nodes (triage, reporting)
    Deterministic tools handle the compute (inference, GradCAM, drift detection)

The graph has 6 nodes:
    classify  →  triage  →  reason  →  explain (conditional)  →  review (conditional)  →  report

Claude decides at the 'triage' and 'report' nodes. Everything else is deterministic.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, END

from src.agents import llm
from src.agents.event_bus import EventType, Event, event_bus

logger = logging.getLogger(__name__)


# ── Graph State ──

class ScreeningState(TypedDict, total=False):
    """State that flows through the LangGraph screening pipeline."""
    # Input
    scan_id: str
    image: Any  # PIL.Image
    threshold: float | None

    # Classification output
    predictions: list[dict]
    all_probabilities: dict
    referral_priority: str
    inference_ms: float

    # Triage decision (Claude or deterministic)
    triage: dict  # {priority, should_explain, should_review, reasoning}

    # Clinical reasoning output
    reasoning: dict  # {adjustments, visual_findings, treatment}

    # Explainability output
    explainability: dict | None

    # Review decision
    review: dict | None

    # Final report (Claude-generated or template)
    report: dict

    # Metadata
    steps_completed: list[str]
    errors: list[str]
    claude_used: bool


# ── Node functions ──

async def classify_node(state: ScreeningState) -> dict:
    """Node 1: Run model inference. Always deterministic (GPU inference)."""
    from backend.app.core.model_service import model_service

    image = state["image"]
    threshold = state.get("threshold")

    result = model_service.predict(image, threshold=threshold)

    return {
        "predictions": result.get("predictions", []),
        "all_probabilities": result.get("all_probabilities", {}),
        "referral_priority": result.get("clinical", {}).get("referral_priority", "FOLLOW_UP"),
        "inference_ms": result.get("inference_ms", 0),
        "steps_completed": state.get("steps_completed", []) + ["classify"],
    }


async def triage_node(state: ScreeningState) -> dict:
    """Node 2: Triage — Claude decides urgency, whether to explain, and whether to review.

    If Claude is unavailable, falls back to rule-based triage.
    """
    predictions = state.get("predictions", [])
    referral = state.get("referral_priority", "FOLLOW_UP")

    CRITICAL = {"CRAO", "AION", "CRVO", "VH", "RS"}
    EMERGENCY = {"CRAO", "AION"}

    detected_codes = [p["code"] for p in predictions]
    has_critical = any(c in CRITICAL for c in detected_codes)
    has_emergency = any(c in EMERGENCY for c in detected_codes)
    low_confidence = any(p.get("probability", 1.0) < 0.70 for p in predictions)

    # Try Claude for nuanced triage
    if llm.is_available() and len(predictions) > 0:
        disease_summary = "\n".join(
            f"- {p['name']} ({p['code']}): {p['probability']:.1%} confidence [{p.get('confidence', 'unknown')}]"
            for p in predictions[:10]
        )

        prompt = f"""Triage this retinal screening result. {len(predictions)} diseases detected.
Referral priority from classifier: {referral}

Detected diseases:
{disease_summary}

Respond with exactly this JSON format (no markdown):
{{"priority": "EMERGENCY|URGENT|ROUTINE|FOLLOW_UP", "should_explain": true/false, "should_review": true/false, "reasoning": "one sentence clinical reasoning"}}"""

        response = await llm.invoke(prompt, max_tokens=200)

        if not response["fallback"]:
            import json
            try:
                # Parse Claude's structured response
                text = response["text"].strip()
                if text.startswith("```"):
                    text = text.split("```")[1].replace("json", "").strip()
                triage = json.loads(text)
                triage["source"] = "claude"
                return {
                    "triage": triage,
                    "claude_used": True,
                    "steps_completed": state.get("steps_completed", []) + ["triage_claude"],
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Claude triage parse failed, using rules: {e}")

    # Deterministic fallback
    triage = {
        "priority": "EMERGENCY" if has_emergency else referral,
        "should_explain": has_critical or len(predictions) >= 3,
        "should_review": low_confidence or len(predictions) > 5 or has_emergency,
        "reasoning": _rule_based_reasoning(predictions, referral, has_critical, has_emergency),
        "source": "rules",
    }

    return {
        "triage": triage,
        "claude_used": False,
        "steps_completed": state.get("steps_completed", []) + ["triage_rules"],
    }


def _rule_based_reasoning(predictions, referral, has_critical, has_emergency) -> str:
    if has_emergency:
        return "Sight-threatening arterial occlusion detected — immediate referral required"
    if has_critical:
        return f"Critical pathology detected among {len(predictions)} findings — specialist review recommended"
    if len(predictions) > 5:
        return f"Complex multi-disease presentation ({len(predictions)} findings) — review for co-management"
    if len(predictions) > 0:
        return f"{len(predictions)} finding(s) detected at {referral} priority"
    return "No significant pathology detected"


async def reason_node(state: ScreeningState) -> dict:
    """Node 3: Apply clinical knowledge graph reasoning. Always deterministic."""
    from backend.app.core.model_service import model_service

    kg = model_service.kg
    if not kg:
        return {
            "reasoning": {"error": "Knowledge graph not loaded"},
            "steps_completed": state.get("steps_completed", []) + ["reason"],
        }

    probs = {}
    for code, data in state.get("all_probabilities", {}).items():
        probs[code] = data["probability"] if isinstance(data, dict) else float(data)

    refined = kg.apply_clinical_reasoning(probs)
    detected = [c for c, p in refined.items() if p > 0.3]
    treatment = kg.get_treatment_recommendations(detected)
    visual_findings = kg.get_visual_findings(detected)

    # Compute adjustments
    adjustments = []
    for code in refined:
        if code in probs:
            diff = refined[code] - probs[code]
            if abs(diff) > 0.001:
                adjustments.append({
                    "code": code,
                    "original": round(probs[code], 4),
                    "refined": round(refined[code], 4),
                    "boost": round(diff, 4),
                })

    return {
        "reasoning": {
            "adjustments": adjustments,
            "treatment_recommendations": treatment,
            "visual_findings": visual_findings,
            "diseases_after_reasoning": len(detected),
        },
        "steps_completed": state.get("steps_completed", []) + ["reason"],
    }


async def explain_node(state: ScreeningState) -> dict:
    """Node 4 (conditional): Generate explainability artifacts."""
    predictions = state.get("predictions", [])
    target_diseases = [p["code"] for p in predictions[:3]]

    # In production this calls the actual GradCAM/LIME pipeline
    # Here we record what would be explained
    return {
        "explainability": {
            "method": "GradCAM",
            "target_diseases": target_diseases,
            "generated": len(target_diseases),
            "note": "Full heatmaps generated via /api/v1/explain/gradcam endpoint",
        },
        "steps_completed": state.get("steps_completed", []) + ["explain"],
    }


async def review_node(state: ScreeningState) -> dict:
    """Node 5 (conditional): Flag for human review."""
    triage = state.get("triage", {})
    scan_id = state.get("scan_id", "unknown")

    # Emit review event
    await event_bus.emit(Event(
        type=EventType.REVIEW_REQUIRED,
        source="screening_graph",
        data={
            "scan_id": scan_id,
            "reason": triage.get("reasoning", ""),
            "priority": "urgent" if triage.get("priority") in ("EMERGENCY", "URGENT") else "medium",
        },
    ))

    return {
        "review": {
            "flagged": True,
            "reason": triage.get("reasoning", ""),
            "priority": triage.get("priority", "ROUTINE"),
        },
        "steps_completed": state.get("steps_completed", []) + ["review"],
    }


async def report_node(state: ScreeningState) -> dict:
    """Node 6: Generate final clinical screening report.

    Claude generates a natural-language clinical summary when available.
    Falls back to a structured template otherwise.
    """
    predictions = state.get("predictions", [])
    triage = state.get("triage", {})
    reasoning = state.get("reasoning", {})
    referral = state.get("referral_priority", "FOLLOW_UP")
    scan_id = state.get("scan_id", "unknown")

    # Try Claude for natural-language report
    clinical_narrative = ""
    if llm.is_available() and len(predictions) > 0:
        disease_list = ", ".join(f"{p['name']} ({p['probability']:.0%})" for p in predictions[:8])
        adjustments = reasoning.get("adjustments", [])
        adj_text = "; ".join(f"{a['code']} boosted +{a['boost']:.1%}" for a in adjustments[:5]) if adjustments else "none"

        prompt = f"""Write a concise clinical screening report (3-4 sentences) for this retinal scan.

Scan ID: {scan_id}
Detected: {disease_list}
Referral: {triage.get('priority', referral)}
KG Adjustments: {adj_text}
Triage reasoning: {triage.get('reasoning', '')}

Write as if you are documenting findings for the referring ophthalmologist.
Be specific about disease codes and probabilities. End with a clear recommendation."""

        response = await llm.invoke(prompt, max_tokens=300)
        if not response["fallback"]:
            clinical_narrative = response["text"]

    # Build structured report
    report = {
        "scan_id": scan_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diseases_detected": len(predictions),
        "referral_priority": triage.get("priority", referral),
        "top_findings": [
            {"code": p["code"], "name": p["name"], "probability": round(p["probability"], 3)}
            for p in predictions[:10]
        ],
        "clinical_narrative": clinical_narrative or _template_narrative(predictions, referral, triage),
        "triage": triage,
        "kg_adjustments": len(reasoning.get("adjustments", [])),
        "explainability_run": state.get("explainability") is not None,
        "flagged_for_review": state.get("review", {}).get("flagged", False),
        "steps_completed": state.get("steps_completed", []) + ["report"],
        "claude_powered": state.get("claude_used", False),
    }

    # Emit final event
    await event_bus.emit(Event(
        type=EventType.SCAN_ANALYZED,
        source="screening_graph",
        data={
            "scan_id": scan_id,
            "diseases_detected": len(predictions),
            "referral_priority": report["referral_priority"],
            "claude_used": state.get("claude_used", False),
            "steps": len(report["steps_completed"]),
        },
    ))

    return {"report": report}


def _template_narrative(predictions, referral, triage) -> str:
    if not predictions:
        return "AI screening found no significant retinal pathology. Routine follow-up recommended."
    top = predictions[0]
    count = len(predictions)
    return (
        f"AI screening identified {count} finding(s). "
        f"Primary finding: {top['name']} ({top['code']}) at {top['probability']:.0%} confidence. "
        f"Referral priority: {triage.get('priority', referral)}. "
        f"{triage.get('reasoning', 'Clinical review recommended.')} "
        f"This is an AI-assisted screening result and requires specialist confirmation."
    )


# ── Routing functions ──

def route_after_reason(state: ScreeningState) -> Literal["explain", "review", "report"]:
    """Route after clinical reasoning: explain, review, or straight to report."""
    triage = state.get("triage", {})
    if triage.get("should_explain", False):
        return "explain"
    if triage.get("should_review", False):
        return "review"
    return "report"


def route_after_explain(state: ScreeningState) -> Literal["review", "report"]:
    """Route after explainability: review or straight to report."""
    triage = state.get("triage", {})
    if triage.get("should_review", False):
        return "review"
    return "report"


# ── Build the graph ──

def build_screening_graph() -> StateGraph:
    """Construct the LangGraph screening workflow.

    Graph topology:
        classify → triage → reason ─┬→ explain ─┬→ review → report → END
                                     │           └→ report → END
                                     ├→ review → report → END
                                     └→ report → END
    """
    graph = StateGraph(ScreeningState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("triage", triage_node)
    graph.add_node("reason", reason_node)
    graph.add_node("explain", explain_node)
    graph.add_node("review", review_node)
    graph.add_node("report", report_node)

    # Linear edges
    graph.add_edge("classify", "triage")
    graph.add_edge("triage", "reason")

    # After reason: 3-way branch (explain, review-only, or report)
    graph.add_conditional_edges("reason", route_after_reason, {
        "explain": "explain",
        "review": "review",
        "report": "report",
    })

    # After explain: review or report
    graph.add_conditional_edges("explain", route_after_explain, {
        "review": "review",
        "report": "report",
    })

    # Review always goes to report
    graph.add_edge("review", "report")

    # Report is terminal
    graph.add_edge("report", END)

    # Entry point
    graph.set_entry_point("classify")

    return graph


def _compile_graph():
    """Compile graph with error handling."""
    try:
        return build_screening_graph().compile()
    except Exception as e:
        logger.error(f"Failed to compile LangGraph screening graph: {e}", exc_info=True)
        return None


screening_graph = _compile_graph()


async def run_screening(image, scan_id: str = "", threshold: float | None = None) -> dict:
    """Execute the full LangGraph screening pipeline.

    This is the main entry point — call this from the API or agent.
    Returns the final report dict.
    """
    if screening_graph is None:
        return {"error": "LangGraph screening graph failed to compile", "status": "error"}

    import uuid

    sid = scan_id or str(uuid.uuid4())[:8]

    initial_state: ScreeningState = {
        "scan_id": sid,
        "image": image,
        "threshold": threshold,
        "steps_completed": [],
        "errors": [],
        "claude_used": False,
    }

    result = await screening_graph.ainvoke(initial_state)

    return result.get("report", {"error": "Graph produced no report"})
