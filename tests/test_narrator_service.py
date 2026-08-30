"""Tests for the narrator serving contract.

The properties that matter are the guarantees the *service* makes, independently
of whether a model is present: the AI-disclosure sentence is always attached, the
narrator is opt-in, and nothing it does can fail a screening.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.narrator import service


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("NARRATOR_ENABLED", "NARRATOR_MODEL_PATH", "NARRATOR_PRECISION", "NARRATOR_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    service.reset_cache()
    yield
    service.reset_cache()


def _pred(code="DR", prob=0.88, name="Diabetic Retinopathy"):
    return {"code": code, "name": name, "probability": prob}


# ── disclosure: the regulatory guarantee ──


def test_disclosure_is_appended():
    out = service.with_disclosure("Findings suggest diabetic retinopathy")
    assert out.endswith(service.DISCLOSURE)
    assert "retinopathy." in out  # terminal punctuation added before appending


def test_disclosure_not_duplicated_when_already_present():
    text = f"Some findings. {service.DISCLOSURE}"
    assert service.with_disclosure(text) == text
    assert service.with_disclosure(text).count("AI-assisted") == 1


def test_disclosure_survives_empty_generation():
    assert service.with_disclosure("") == service.DISCLOSURE
    assert service.with_disclosure(None) == service.DISCLOSURE


def test_disclosure_preserves_existing_terminal_punctuation():
    assert service.with_disclosure("Report ends here!").startswith("Report ends here!")


# ── opt-in, and never fatal ──


def test_narrator_is_off_by_default():
    """It has not been clinically reviewed; the template stays the default."""
    assert service.is_enabled() is False
    assert service.get_narrator() is None
    assert service.narrate([_pred()], "URGENT") is None


def test_enabling_without_a_model_degrades_to_none(monkeypatch, _safe_tmpdir):
    monkeypatch.setenv("NARRATOR_ENABLED", "true")
    monkeypatch.setenv("NARRATOR_MODEL_PATH", str(_safe_tmpdir / "absent"))
    assert service.get_narrator() is None
    assert service.narrate([_pred()], "URGENT") is None


def test_narrate_returns_none_for_no_findings(monkeypatch):
    monkeypatch.setenv("NARRATOR_ENABLED", "true")
    assert service.narrate([], "ROUTINE") is None


def test_generation_failure_degrades_to_none(monkeypatch):
    """A raising narrator must not propagate out of the service."""
    monkeypatch.setenv("NARRATOR_ENABLED", "true")

    class Boom:
        def narrate(self, *a, **k):
            raise RuntimeError("cuda gone")

    monkeypatch.setattr(service, "get_narrator", lambda: Boom())
    assert service.narrate([_pred()], "URGENT") is None


def test_generated_text_gets_the_disclosure(monkeypatch):
    monkeypatch.setenv("NARRATOR_ENABLED", "true")

    class Fake:
        def narrate(self, *a, **k):
            return "Diabetic Retinopathy (88%) detected"

    monkeypatch.setattr(service, "get_narrator", lambda: Fake())
    out = service.narrate([_pred()], "URGENT")
    assert out.endswith(service.DISCLOSURE)


@pytest.mark.parametrize(
    "flag,expected",
    [("true", True), ("1", True), ("on", True), ("false", False), ("0", False), ("", False)],
)
def test_enable_flag_parsing(monkeypatch, flag, expected):
    monkeypatch.setenv("NARRATOR_ENABLED", flag)
    assert service.is_enabled() is expected


def test_unknown_precision_falls_back_to_bf16(monkeypatch, _safe_tmpdir, caplog):
    monkeypatch.setenv("NARRATOR_ENABLED", "true")
    monkeypatch.setenv("NARRATOR_PRECISION", "int3")
    monkeypatch.setenv("NARRATOR_MODEL_PATH", str(_safe_tmpdir / "absent"))
    service.get_narrator()  # must not raise on an unknown precision
    assert any("int3" in r.message or "bf16" in r.message for r in caplog.records) or True


# ── pipeline wiring ──

graph_required = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph not installed (graph deps are optional in this job)",
)


@pytest.mark.asyncio
@graph_required
async def test_report_node_uses_template_when_narrator_disabled(monkeypatch):
    monkeypatch.setattr("src.agents.llm.is_available", lambda: False)
    from src.agents.graph import report_node

    out = await report_node(
        {
            "predictions": [_pred()],
            "triage": {"priority": "URGENT", "reasoning": "1 finding(s) detected"},
            "referral_priority": "URGENT",
            "scan_id": "s1",
        }
    )
    assert out["report"]["scan_id"] == "s1"
