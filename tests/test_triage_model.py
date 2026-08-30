"""Tests for the lightweight learned triage head and its pipeline wiring.

Two properties matter most and neither is covered by the offline sweep:

* the **deterministic emergency override** — no EMERGENCY case appeared in the
  240 real traces, so escalation is only ever exercised here;
* **graceful degradation** — a missing or malformed artifact must fall back to
  the existing deterministic behaviour, never fail a screening.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.triage import features as tf
from src.triage.model import (
    DEFAULT_MODEL_PATH,
    TriageModel,
    get_model,
    is_enabled,
    reset_cache,
)

ARTIFACT = Path(__file__).resolve().parents[1] / DEFAULT_MODEL_PATH


@pytest.fixture(autouse=True)
def _clean_model_cache(monkeypatch):
    """Each test gets a fresh loader state and a predictable environment."""
    monkeypatch.delenv("TRIAGE_MODEL_ENABLED", raising=False)
    monkeypatch.delenv("TRIAGE_MODEL_PATH", raising=False)
    reset_cache()
    yield
    reset_cache()


def _pred(code, prob, name=None):
    return {"code": code, "name": name or code, "probability": prob}


# ── feature encoder ──


def test_feature_vector_layout_matches_names():
    vec = tf.encode([("DR", 0.9)], "URGENT")
    assert len(vec) == len(tf.FEATURE_NAMES) == 27


def test_duplicate_codes_collapse_to_max_probability():
    names = tf.feature_names()
    dr = names.index("p_DR")
    assert tf.encode([("DR", 0.4), ("DR", 0.8)], "ROUTINE")[dr] == pytest.approx(0.8)


def test_referral_is_one_hot():
    names = tf.feature_names()
    vec = tf.encode([("DR", 0.9)], "ROUTINE")
    hot = [vec[names.index(f"referral_{p}")] for p in tf.PRIORITIES]
    assert hot == [0.0, 0.0, 1.0, 0.0]


def test_empty_findings_do_not_divide_by_zero():
    vec = tf.encode([], "FOLLOW_UP")
    names = tf.feature_names()
    assert vec[names.index("mean_prob")] == 0.0
    assert vec[names.index("max_prob")] == 0.0
    assert vec[names.index("n_findings")] == 0.0


def test_evaluation_encoder_delegates_to_production_encoder():
    """The fitted-on and served-on vectors must be the same vector."""
    from src.evaluation.reasoner_comparison.features import case_features
    from src.evaluation.reasoner_comparison.interface import Case, Prediction

    case = Case(
        scan_id="c1",
        predictions=[
            Prediction("DR", "Diabetic Retinopathy", 0.81),
            Prediction("CRVO", "Central Retinal Vein Occlusion", 0.62),
        ],
        probabilities={},
        referral_priority="URGENT",
    )
    assert case_features(case) == tf.encode([("DR", 0.81), ("CRVO", 0.62)], "URGENT")


# ── shipped artifact ──


@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
def test_shipped_artifact_loads_and_is_self_describing():
    m = TriageModel.load(ARTIFACT)
    assert len(m.coef[0]) == len(tf.FEATURE_NAMES)
    assert m.disease_codes  # vocabulary travels with the weights
    # EMERGENCY is deliberately absent from the learned classes.
    assert "EMERGENCY" not in [m.priorities[c] for c in m.classes]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
def test_emergency_code_always_escalates_even_though_never_learned():
    m = TriageModel.load(ARTIFACT)
    for code in ("CRAO", "AION"):
        # low probability and the calmest possible referral — must still escalate
        d = m.decide([_pred(code, 0.31)], "ROUTINE")
        assert d.priority == "EMERGENCY"
        assert d.should_review is True
        assert "override" in d.source


@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
def test_non_emergency_case_uses_the_learned_head():
    m = TriageModel.load(ARTIFACT)
    d = m.decide([_pred("DR", 0.88)], "URGENT")
    assert d.priority in tf.PRIORITIES
    assert d.source == "triage_model"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
def test_review_and_explain_flags_follow_the_rule_baseline():
    m = TriageModel.load(ARTIFACT)
    assert (
        m.decide([_pred("DR", 0.9), _pred("ARMD", 0.8), _pred("MH", 0.8)], "URGENT").should_explain
        is True
    )
    assert m.decide([_pred("DR", 0.55)], "FOLLOW_UP").should_review is True  # low confidence


# ── loader contract: never take down a screening ──


def test_missing_artifact_degrades_to_none(monkeypatch, _safe_tmpdir):
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(_safe_tmpdir / "absent.json"))
    assert get_model() is None


def test_malformed_artifact_degrades_to_none(monkeypatch, _safe_tmpdir):
    bad = _safe_tmpdir / "bad.json"
    bad.write_text("{not json")
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(bad))
    assert get_model() is None


def test_unknown_format_is_refused():
    with pytest.raises(ValueError, match="unsupported triage model format"):
        TriageModel({"format": "something-else"})


def test_feature_layout_drift_is_refused():
    """A model whose recorded columns disagree with the encoder must not serve."""
    spec = {
        "format": "linear-softmax-v1",
        "priorities": list(tf.PRIORITIES),
        "classes": [1],
        "disease_codes": ["DR", "ARMD"],
        "feature_names": ["p_DR", "p_ARMD", "stale_column"],  # not what encode() yields
        "coef": [[0.0, 0.0, 0.0]],
        "intercept": [0.0],
    }
    with pytest.raises(ValueError, match="feature layout changed"):
        TriageModel(spec)


def test_env_flag_disables_the_head(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_ENABLED", "false")
    assert is_enabled() is False
    assert get_model() is None


@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
def test_enabled_by_default(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(ARTIFACT))
    assert is_enabled() is True
    assert get_model() is not None


# ── pipeline wiring ──
#
# These exercise ``src.agents.graph``, which pulls in langgraph. The triage head
# itself has no such dependency (pure stdlib), and some CI jobs install only the
# lighter test extras — so skip rather than fail when the graph deps are absent.

graph_required = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph not installed (graph deps are optional in this job)",
)


@pytest.mark.asyncio
@graph_required
@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
async def test_triage_node_uses_the_head_without_an_llm(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(ARTIFACT))
    from src.agents.graph import triage_node

    out = await triage_node(
        {
            "predictions": [_pred("DR", 0.88, "Diabetic Retinopathy")],
            "referral_priority": "URGENT",
        }
    )
    assert out["triage"]["source"].startswith("triage_model")
    assert out["claude_used"] is False
    assert "triage_model" in out["steps_completed"]


@pytest.mark.asyncio
@graph_required
@pytest.mark.skipif(not ARTIFACT.exists(), reason="triage artifact not built")
async def test_triage_node_escalates_emergency(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(ARTIFACT))
    from src.agents.graph import triage_node

    out = await triage_node(
        {
            "predictions": [_pred("CRAO", 0.42, "Central Retinal Artery Occlusion")],
            "referral_priority": "ROUTINE",
        }
    )
    assert out["triage"]["priority"] == "EMERGENCY"


@pytest.mark.asyncio
@graph_required
async def test_triage_node_falls_back_to_rules_when_head_disabled(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_ENABLED", "false")
    monkeypatch.setattr("src.agents.llm.is_available", lambda: False)
    from src.agents.graph import triage_node

    out = await triage_node(
        {
            "predictions": [_pred("DR", 0.88, "Diabetic Retinopathy")],
            "referral_priority": "URGENT",
        }
    )
    assert out["triage"]["source"] == "rules"
    assert "triage_rules" in out["steps_completed"]
