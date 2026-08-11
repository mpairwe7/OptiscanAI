"""Tests for the clinical-reasoner comparison harness.

Fast tests cover the metric math, synthetic teacher logic, and the rule/CNN
reasoner contracts. The end-to-end runner + CNN micro-train is marked ``slow``.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reasoner_comparison.cases import (
    CODE_TO_NAME,
    make_synthetic_cases,
    synthetic_teacher,
)
from src.evaluation.reasoner_comparison.interface import (
    PRIORITIES,
    Case,
    Prediction,
    ReasonerOutput,
)
from src.evaluation.reasoner_comparison.metrics import (
    evaluate_gates,
    narrative_metrics,
    ops_metrics,
    triage_metrics,
)
from src.evaluation.reasoner_comparison.reasoners import (
    RuleReasoner,
    _extract_json,
    _rule_triage,
)


def _out(priority, explain=False, review=False, narrative=""):
    return ReasonerOutput(
        priority=priority, should_explain=explain, should_review=review, narrative=narrative
    )


# ── interface ──


def test_reasoner_output_rejects_unknown_priority():
    with pytest.raises(ValueError):
        ReasonerOutput(priority="SOON", should_explain=False, should_review=False)


def test_case_detected_codes_and_summary():
    case = Case(
        scan_id="c1",
        predictions=[Prediction("DR", "Diabetic Retinopathy", 0.9)],
        probabilities={"DR": 0.9},
        referral_priority="ROUTINE",
    )
    assert case.detected_codes == ["DR"]
    assert "DR" in case.disease_summary()


# ── triage metrics ──


def test_triage_metrics_perfect_agreement():
    refs = [_out("EMERGENCY"), _out("ROUTINE"), _out("FOLLOW_UP")]
    m = triage_metrics(refs, list(refs))
    assert m["priority_accuracy"] == 1.0
    assert m["priority_macro_f1"] == pytest.approx(1.0)
    assert m["emergency_recall"] == 1.0
    assert m["cohen_kappa"] == pytest.approx(1.0)


def test_triage_metrics_emergency_miss_is_caught():
    refs = [_out("EMERGENCY"), _out("EMERGENCY"), _out("ROUTINE")]
    preds = [_out("EMERGENCY"), _out("ROUTINE"), _out("ROUTINE")]  # missed one emergency
    m = triage_metrics(refs, preds)
    assert m["emergency_recall"] == pytest.approx(0.5)
    assert m["per_class"]["EMERGENCY"]["support"] == 2


def test_triage_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        triage_metrics([_out("ROUTINE")], [_out("ROUTINE"), _out("ROUTINE")])


def test_binary_flag_f1_present():
    refs = [_out("ROUTINE", explain=True), _out("ROUTINE", explain=False)]
    preds = [_out("ROUTINE", explain=True), _out("ROUTINE", explain=True)]
    m = triage_metrics(refs, preds)
    assert 0.0 <= m["should_explain_f1"] <= 1.0


# ── narrative metrics ──


def test_narrative_grounding_penalizes_hallucination():
    case = Case(
        scan_id="c1",
        predictions=[Prediction("DR", "Diabetic Retinopathy", 0.9)],
        probabilities={"DR": 0.9},
    )
    grounded = [_out("ROUTINE", narrative="Findings: Diabetic Retinopathy (DR) noted.")]
    halluc = [_out("ROUTINE", narrative="Diabetic Retinopathy (DR) and Macular Hole (MH) seen.")]
    code_map = {"DR": "Diabetic Retinopathy", "MH": "Macular Hole"}
    assert narrative_metrics([case], grounded, code_map)["grounding"] == 1.0
    # MH mentioned but not detected -> grounding drops to 0.5
    assert narrative_metrics([case], halluc, code_map)["grounding"] == pytest.approx(0.5)


def test_narrative_empty_rate_for_non_narrating():
    case = Case(
        scan_id="c1",
        predictions=[Prediction("DR", "Diabetic Retinopathy", 0.9)],
        probabilities={"DR": 0.9},
    )
    m = narrative_metrics([case], [_out("ROUTINE", narrative="")], {"DR": "Diabetic Retinopathy"})
    assert m["empty_rate"] == 1.0


def test_narrative_grounding_is_one_when_no_disease_mentions():
    # No-pathology case + narrative naming no diseases = nothing hallucinated.
    case = Case(scan_id="c1", predictions=[], probabilities={})
    out = _out("FOLLOW_UP", narrative="No significant pathology. Routine follow-up advised.")
    m = narrative_metrics([case], [out], {"DR": "Diabetic Retinopathy"})
    assert m["grounding"] == 1.0


# ── gates ──


def test_gate_fails_on_missed_emergency():
    triage = {"emergency_recall": 0.8, "emergency_support": 5, "priority_macro_f1": 0.9}
    ops = {"size_mb": 6.0, "latency_p95_ms": 12.0, "generates_narrative": False}
    gate = evaluate_gates(triage, {}, ops)
    assert gate["passed"] is False
    assert gate["checks"]["emergency_recall"]["pass"] is False


def test_gate_skips_emergency_when_no_emergencies_in_split():
    triage = {"emergency_recall": 0.0, "emergency_support": 0, "priority_macro_f1": 0.9}
    ops = {"size_mb": 6.0, "latency_p95_ms": 12.0, "generates_narrative": False}
    gate = evaluate_gates(triage, {}, ops)
    assert gate["checks"]["emergency_recall"]["pass"] is True


def test_gate_narrative_check_only_when_narrating():
    triage = {"emergency_recall": 1.0, "priority_macro_f1": 0.9}
    ops_no_narr = {"size_mb": 6.0, "latency_p95_ms": 12.0, "generates_narrative": False}
    assert "grounding" not in evaluate_gates(triage, {"grounding": 0.1}, ops_no_narr)["checks"]
    ops_narr = {**ops_no_narr, "generates_narrative": True}
    assert "grounding" in evaluate_gates(triage, {"grounding": 0.1}, ops_narr)["checks"]


def test_server_gate_lifts_size_and_latency_but_keeps_safety():
    from src.evaluation.reasoner_comparison.metrics import SERVER_GATES

    # A heavy, slow, but grounded + emergency-safe narrator: fails edge, passes server.
    triage = {"emergency_recall": 1.0, "emergency_support": 2, "priority_macro_f1": 1.0}
    narrative = {"grounding": 1.0}
    ops = {"size_mb": 988.0, "latency_p95_ms": 6935.0, "generates_narrative": True}
    assert evaluate_gates(triage, narrative, ops)["passed"] is False  # edge caps
    assert evaluate_gates(triage, narrative, ops, SERVER_GATES)["passed"] is True
    # server still enforces grounding + emergency recall
    bad = evaluate_gates(triage, {"grounding": 0.1}, ops, SERVER_GATES)
    assert bad["passed"] is False


def test_ops_metrics_percentiles():
    preds = [_out("ROUTINE") for _ in range(4)]
    for i, p in enumerate(preds):
        p.latency_ms = float(i)  # 0,1,2,3
    info = {"size_mb": 1.0, "offline": True, "generates_narrative": False, "extra_deps": []}
    ops = ops_metrics(info, preds)
    assert ops["latency_p50_ms"] == pytest.approx(1.5)


# ── synthetic teacher + rule reasoner ──


def test_synthetic_teacher_escalates_emergency():
    preds = [Prediction("CRAO", "Central Retinal Artery Occlusion", 0.95)]
    out = synthetic_teacher(preds, referral="URGENT")
    assert out.priority == "EMERGENCY"
    assert out.should_review is True


def test_synthetic_teacher_deescalates_single_low_conf():
    preds = [Prediction("DN", "Drusen", 0.55)]  # non-critical, low conf, single
    out = synthetic_teacher(preds, referral="ROUTINE")
    assert out.priority == "FOLLOW_UP"


def test_rule_reasoner_matches_rule_triage_and_is_valid():
    case = Case(
        scan_id="c1",
        predictions=[Prediction("CRVO", "Central Retinal Vein Occlusion", 0.92)],
        probabilities={"CRVO": 0.92},
        referral_priority="URGENT",
    )
    out = RuleReasoner().reason(case)
    priority, explain, review, _ = _rule_triage(case)
    assert out.priority == priority in PRIORITIES
    assert out.source == "rule_baseline"
    assert out.narrative  # template narrative is grounded/non-empty
    assert out.latency_ms >= 0.0


def test_make_synthetic_cases_have_references_and_images():
    cases = make_synthetic_cases(n=12, seed=7, img_size=32)
    assert len(cases) == 12
    assert all(c.reference is not None for c in cases)
    assert all(c.reference.priority in PRIORITIES for c in cases)
    assert all(tuple(c.image.shape) == (3, 32, 32) for c in cases)
    assert set(CODE_TO_NAME).issuperset(set(cases[0].probabilities))


# ── feature-based triage ──


def _case(codes_probs, referral="ROUTINE"):
    preds = [Prediction(c, CODE_TO_NAME.get(c, c), p) for c, p in codes_probs]
    return Case(
        scan_id="c",
        predictions=preds,
        probabilities={c: p for c, p in codes_probs},
        referral_priority=referral,
    )


def test_case_features_layout_and_referral_toggle():
    from src.evaluation.reasoner_comparison.features import (
        case_features,
        feature_names,
    )

    case = _case([("DR", 0.9), ("CRAO", 0.95)], referral="URGENT")
    full = case_features(case, include_referral=True)
    noref = case_features(case, include_referral=False)
    assert len(full) == len(feature_names(include_referral=True))
    assert len(noref) == len(feature_names(include_referral=False))
    assert len(full) - len(noref) == len(PRIORITIES)  # one-hot referral block
    # emergency + critical flags fire on CRAO; n_findings is 2
    names = feature_names(include_referral=True)
    assert full[names.index("has_emergency")] == 1.0
    assert full[names.index("has_critical")] == 1.0
    assert full[names.index("n_findings")] == 2.0
    assert full[names.index("referral_URGENT")] == 1.0


def test_priority_classifier_returns_priority_indices():
    """PriorityClassifier must round-trip non-contiguous PRIORITY_INDEX labels."""
    from sklearn.tree import DecisionTreeClassifier

    from src.evaluation.reasoner_comparison.features import PriorityClassifier
    from src.evaluation.reasoner_comparison.interface import PRIORITY_INDEX

    # labels use only URGENT(1)/ROUTINE(2)/FOLLOW_UP(3) — EMERGENCY(0) absent.
    feats = [[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]]
    y = [PRIORITY_INDEX["URGENT"]] * 2 + [PRIORITY_INDEX["ROUTINE"]] * 2 + [
        PRIORITY_INDEX["FOLLOW_UP"]
    ] * 2
    clf = PriorityClassifier(DecisionTreeClassifier(random_state=0)).fit(feats, y)
    preds = clf.predict([[0.0], [2.0]])
    assert int(preds[0]) == PRIORITY_INDEX["URGENT"]
    assert int(preds[1]) == PRIORITY_INDEX["FOLLOW_UP"]


class _StubEstimator:
    """Always predicts a fixed PRIORITY_INDEX — isolates reasoner wiring."""

    def __init__(self, idx):
        self.idx = idx

    def predict(self, feats):
        return [self.idx for _ in feats]


def test_feature_triage_reasoner_emergency_override():
    from src.evaluation.reasoner_comparison.interface import PRIORITY_INDEX
    from src.evaluation.reasoner_comparison.reasoners import FeatureTriageReasoner

    # estimator insists on ROUTINE; an emergency code must escalate to EMERGENCY.
    reasoner = FeatureTriageReasoner(_StubEstimator(PRIORITY_INDEX["ROUTINE"]), name="feat_test")
    no_emerg = reasoner.reason(_case([("DR", 0.9)], referral="ROUTINE"))
    assert no_emerg.priority == "ROUTINE"  # learned head respected
    assert no_emerg.source == "feat_test"
    assert no_emerg.narrative  # grounded template
    with_emerg = reasoner.reason(_case([("CRAO", 0.95)], referral="ROUTINE"))
    assert with_emerg.priority == "EMERGENCY"  # safety override wins
    assert reasoner.size_mb() >= 0.0


# ── end-to-end (slow: trains a tiny CNN) ──


@pytest.mark.slow
def test_smoke_runner_writes_report(_safe_tmpdir):
    from src.evaluation.reasoner_comparison.cnn import train_triage_cnn
    from src.evaluation.reasoner_comparison.reasoners import CNNTriageReasoner
    from src.evaluation.reasoner_comparison.runner import run_comparison

    cases = make_synthetic_cases(n=40, seed=3, img_size=32)
    train, test = cases[:28], cases[28:]
    model = train_triage_cnn(train, epochs=1, img_size=32, device="cpu")
    reasoners = [RuleReasoner(), CNNTriageReasoner(model, img_size=32)]
    payload = run_comparison(
        test,
        reasoners,
        CODE_TO_NAME,
        _safe_tmpdir,
        mode="smoke",
        teacher_source="synthetic_teacher",
    )
    assert (_safe_tmpdir / "report.md").exists()
    assert (_safe_tmpdir / "results.json").exists()
    assert set(payload["reasoners"]) == {"rule_baseline", "cnn_triage"}
    for r in payload["reasoners"].values():
        assert "gate" in r and "passed" in r["gate"]


def test_distilled_reasoner_fails_cleanly():
    """DistilledLLMReasoner must fail loudly, never import at module load.

    Without ``transformers`` it raises ImportError; with it, a bogus checkpoint
    dir raises at load — either way construction never silently succeeds.
    """
    from src.evaluation.reasoner_comparison.reasoners import DistilledLLMReasoner

    try:
        import transformers  # noqa: F401

        has_transformers = True
    except ImportError:
        has_transformers = False

    if has_transformers:
        with pytest.raises(Exception):
            DistilledLLMReasoner("nonexistent-model-dir-xyz-123")
    else:
        with pytest.raises(ImportError):
            DistilledLLMReasoner("nonexistent-model-dir-xyz-123")


# ── distilled-narrator output parsing (regression) ──


def test_extract_json_parses_object_and_strips_code_fence():
    assert _extract_json('{"priority": "URGENT"}')["priority"] == "URGENT"
    fenced = '```json\n{"priority": "ROUTINE"}\n```'
    assert _extract_json(fenced)["priority"] == "ROUTINE"


@pytest.mark.parametrize("payload", ['"just a string"', "42", "[1, 2, 3]", "null"])
def test_extract_json_rejects_valid_json_that_is_not_an_object(payload):
    """A degraded small model can emit valid JSON that isn't a dict.

    That used to reach ``t.get(...)`` and raise AttributeError, killing the whole
    sweep. It must surface as JSONDecodeError so callers fall back to the rule
    seed instead of crashing.
    """
    with pytest.raises(json.JSONDecodeError):
        _extract_json(payload)


def test_reasoner_output_tracks_whether_narrative_was_generated():
    """``grounding`` is 1.0 by construction for the template fallback, so the
    generated/fallback distinction is what makes a failing narrator visible."""
    assert _out("URGENT").narrative_generated is False
    generated = ReasonerOutput(
        priority="URGENT", should_explain=True, should_review=True,
        narrative="model text", narrative_generated=True,
    )
    assert generated.narrative_generated is True
