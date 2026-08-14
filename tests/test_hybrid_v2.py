"""Tests for RetinalFoundationHybridV2 and precision rescue components."""

import os
import json
import tempfile
import shutil

import numpy as np
import pytest
import torch

os.environ["USE_PRETRAINED"] = "0"
os.environ["FAST_SINGLE_RESOLUTION"] = "1"


@pytest.fixture
def knowledge_graph():
    from src.models.vignn import ClinicalKnowledgeGraph
    names = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM"]
    return ClinicalKnowledgeGraph(disease_names=names)


@pytest.fixture
def model_v2(knowledge_graph):
    from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
    return RetinalFoundationHybridV2(
        num_classes=8,
        hidden_dim=128,
        clinical_knowledge_graph=knowledge_graph,
        backbone="vit_small_patch16_224",
        use_lora=False,
        freeze_backbone=False,
        head_dropout1=0.5,
        head_dropout2=0.3,
    )


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 224, 224)


# --- Model tests ---

class TestHybridV2Forward:
    def test_forward_shape(self, model_v2, dummy_input):
        logits = model_v2(dummy_input)
        assert logits.shape == (2, 8)

    def test_predict_returns_dict(self, model_v2, dummy_input):
        result = model_v2.predict(dummy_input)
        assert "logits" in result
        assert "probabilities" in result
        assert "predictions" in result
        assert result["probabilities"].min() >= 0
        assert result["probabilities"].max() <= 1

    def test_predict_with_tta(self, model_v2, dummy_input):
        result = model_v2.predict_with_tta(dummy_input[:1], n_augments=3)
        assert "probabilities" in result
        assert result["probabilities"].shape == (1, 8)

    def test_gradient_flow(self, model_v2, dummy_input):
        logits = model_v2(dummy_input)
        loss = logits.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model_v2.parameters() if p.requires_grad
        )
        assert has_grad

    def test_clinical_reasoning(self, model_v2, dummy_input):
        result = model_v2.predict_with_clinical_reasoning(
            dummy_input[:1], use_tta=False
        )
        assert "predictions" in result
        assert "detected_diseases" in result
        assert "referral_priority" in result
        assert result["referral_priority"] in ("URGENT", "ROUTINE", "FOLLOW_UP")

    def test_knowledge_graph_required(self):
        from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
        with pytest.raises(ValueError):
            RetinalFoundationHybridV2(clinical_knowledge_graph=None)


# --- Staged unfreezing ---

class TestStagedUnfreezing:
    def test_unfreeze_returns_param_groups(self, knowledge_graph):
        """Test with frozen backbone so there are params to unfreeze."""
        from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
        model = RetinalFoundationHybridV2(
            num_classes=8,
            hidden_dim=128,
            clinical_knowledge_graph=knowledge_graph,
            backbone="vit_small_patch16_224",
            use_lora=False,
            freeze_backbone=True,
            head_dropout1=0.5,
            head_dropout2=0.3,
        )
        groups = model.unfreeze_backbone_blocks(num_blocks=2, lr=1e-6)
        assert len(groups) == 1
        assert "params" in groups[0]
        assert groups[0]["lr"] == 1e-6
        unfrozen = sum(p.numel() for p in groups[0]["params"])
        assert unfrozen > 0


# --- Bottleneck Classifier ---

class TestBottleneckClassifier:
    def test_output_shape(self):
        from src.models.retinal_foundation_hybrid_v2 import BottleneckClassifier
        head = BottleneckClassifier(input_dim=512, num_classes=28)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 28)

    def test_dropout_effect(self):
        from src.models.retinal_foundation_hybrid_v2 import BottleneckClassifier
        head = BottleneckClassifier(input_dim=256, num_classes=10, dropout1=0.5, dropout2=0.3)
        x = torch.randn(8, 256)
        head.train()
        out1 = head(x)
        out2 = head(x)
        # With 0.5 dropout, outputs should differ in training mode
        assert not torch.allclose(out1, out2)


# --- Asymmetric Loss V2 ---

class TestASLV2:
    def test_asl_computes(self):
        from src.models.retinal_foundation_hybrid_v2 import AsymmetricLossV2
        loss_fn = AsymmetricLossV2(gamma_neg=4.0, gamma_pos=0.0)
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0
        targets[1, 3] = 1.0
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_asl_gradient(self):
        from src.models.retinal_foundation_hybrid_v2 import AsymmetricLossV2
        loss_fn = AsymmetricLossV2()
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_asl_penalizes_false_positives(self):
        from src.models.retinal_foundation_hybrid_v2 import AsymmetricLossV2
        loss_fn = AsymmetricLossV2(gamma_neg=4.0, gamma_pos=0.0)
        targets = torch.zeros(1, 5)  # All negative

        # High false positive logits should produce higher loss
        fp_logits = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0]])
        tn_logits = torch.tensor([[-3.0, -3.0, -3.0, -3.0, -3.0]])

        fp_loss = loss_fn(fp_logits, targets)
        tn_loss = loss_fn(tn_logits, targets)
        assert fp_loss > tn_loss


# --- Threshold Optimizer ---

class TestThresholdOptimizer:
    def test_optimize_basic(self):
        from src.evaluation.precision_threshold_optimizer import (
            optimize_thresholds_with_precision_floor,
        )
        np.random.seed(42)
        n, c = 200, 5
        probs = np.random.rand(n, c)
        labels = (np.random.rand(n, c) > 0.8).astype(np.float32)

        thresholds, report = optimize_thresholds_with_precision_floor(
            probs, labels, min_precision=0.10
        )

        assert thresholds.shape == (c,)
        assert "summary" in report
        assert "per_class" in report
        assert all(0.0 <= t <= 1.0 for t in thresholds)

    def test_precision_floor_respected(self):
        from src.evaluation.precision_threshold_optimizer import (
            optimize_thresholds_with_precision_floor,
        )
        np.random.seed(42)
        n, c = 500, 3
        probs = np.random.rand(n, c)
        labels = (np.random.rand(n, c) > 0.7).astype(np.float32)

        thresholds, report = optimize_thresholds_with_precision_floor(
            probs, labels, min_precision=0.15
        )

        # Check that optimized classes meet precision floor
        for name, info in report["per_class"].items():
            if info["status"] == "optimized":
                assert info["precision"] >= 0.15 - 0.01  # Small tolerance

    def test_save_thresholds(self):
        from src.evaluation.precision_threshold_optimizer import save_thresholds
        tmpdir = tempfile.mkdtemp(dir="/tmp")
        try:
            thresholds = np.array([0.3, 0.5, 0.7])
            report = {"summary": {"mean_threshold": 0.5}, "per_class": {}}
            path = save_thresholds(thresholds, report, os.path.join(tmpdir, "thresh.json"))
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "thresholds" in data
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# --- Threshold loading/saving on model ---

class TestThresholdManagement:
    def test_load_save_thresholds(self, model_v2):
        tmpdir = tempfile.mkdtemp(dir="/tmp")
        try:
            # Set custom thresholds
            model_v2.thresholds.fill_(0.7)
            path = os.path.join(tmpdir, "thresh.json")
            model_v2.save_thresholds(path)

            # Reset and reload
            model_v2.thresholds.fill_(0.5)
            model_v2.load_thresholds(path)
            assert torch.allclose(model_v2.thresholds, torch.tensor([0.7] * 8))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# --- Fundus Gate ---

class TestLearnedFundusGate:
    def test_gate_forward(self):
        from src.data.fundus_gate_learned import LearnedFundusGate
        gate = LearnedFundusGate(weights_path=None, threshold=0.5)
        x = torch.randn(1, 3, 224, 224)
        out = gate(x)
        assert out.shape == (1, 1)

    def test_gate_check(self):
        from src.data.fundus_gate_learned import LearnedFundusGate
        from PIL import Image
        gate = LearnedFundusGate(weights_path=None)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        is_fundus, confidence, message = gate.check(img)
        assert isinstance(is_fundus, bool)
        assert 0 <= confidence <= 1
        assert isinstance(message, str)


# --- Class filtering ---

class TestAccuracyMetrics:
    def test_accuracy_metrics_present(self):
        from src.training.metrics import compute_multilabel_metrics
        targets = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
        probs = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1], [0.7, 0.6, 0.3], [0.1, 0.2, 0.7]], dtype=np.float32)
        metrics = compute_multilabel_metrics(targets, probs, threshold=0.5)
        assert "accuracy_macro" in metrics
        assert "accuracy_micro" in metrics
        assert "accuracy_jaccard" in metrics
        assert "accuracy_subset" in metrics
        assert 0 <= metrics["accuracy_macro"] <= 1
        assert 0 <= metrics["accuracy_micro"] <= 1
        assert 0 <= metrics["accuracy_jaccard"] <= 1

    def test_perfect_accuracy(self):
        from src.training.metrics import compute_multilabel_metrics
        targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        probs = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], dtype=np.float32)
        metrics = compute_multilabel_metrics(targets, probs, threshold=0.5)
        assert metrics["accuracy_macro"] == 1.0
        assert metrics["accuracy_micro"] == 1.0
        assert metrics["accuracy_subset"] == 1.0


class TestClassFiltering:
    def test_filter_rare_classes(self):
        import pandas as pd
        from src.models.retinal_foundation_hybrid_v2 import filter_rare_classes

        df = pd.DataFrame({
            "DR": [1]*20 + [0]*80,
            "ARMD": [1]*5 + [0]*95,
            "MH": [1]*15 + [0]*85,
            "RARE": [1]*2 + [0]*98,
        })
        kept = filter_rare_classes(["DR", "ARMD", "MH", "RARE"], df, min_samples=10)
        assert "DR" in kept
        assert "MH" in kept
        assert "ARMD" not in kept
        assert "RARE" not in kept


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------


def test_calibration_defaults_to_identity(model_v2, dummy_input):
    """An uncalibrated checkpoint must behave exactly as it did before."""
    assert not model_v2.is_calibrated
    logits = model_v2(dummy_input)
    assert torch.allclose(
        model_v2.calibrated_probabilities(logits), torch.sigmoid(logits), atol=1e-6
    )


def test_set_calibration_applies_platt_scaling(model_v2, dummy_input):
    scale = torch.linspace(1.0, 2.0, model_v2.num_classes)
    shift = torch.linspace(-2.0, -1.0, model_v2.num_classes)
    model_v2.set_calibration(scale, shift)

    assert model_v2.is_calibrated
    logits = model_v2(dummy_input)
    expected = torch.sigmoid(logits * scale + shift)
    assert torch.allclose(model_v2.calibrated_probabilities(logits), expected, atol=1e-6)

    # calibration must lower the probabilities of an over-confident model
    assert model_v2.calibrated_probabilities(logits).mean() < torch.sigmoid(logits).mean()


def test_set_calibration_rejects_wrong_shape(model_v2):
    with pytest.raises(ValueError, match="shape mismatch"):
        model_v2.set_calibration(torch.ones(3), torch.zeros(3))


def test_predict_uses_calibrated_probabilities(model_v2, dummy_input):
    model_v2.eval()
    before = model_v2.predict(dummy_input)["probabilities"]
    model_v2.set_calibration(
        torch.full((model_v2.num_classes,), 1.5),
        torch.full((model_v2.num_classes,), -3.0),
    )
    after = model_v2.predict(dummy_input)["probabilities"]
    assert not torch.allclose(before, after)
    assert (after < before).all()


def test_load_calibration_round_trips(model_v2, dummy_input):
    """The artefact written by scripts/ncc2026_calibration.py must load intact."""
    n = model_v2.num_classes
    payload = {
        "policy": "test",
        "calibration": {
            "scale": [1.0 + 0.1 * i for i in range(n)],
            "shift": [-1.0 - 0.1 * i for i in range(n)],
        },
        "thresholds": [0.2] * n,
    }
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "calibration_artifact.json")
        with open(path, "w") as fh:
            json.dump(payload, fh)
        model_v2.load_calibration(path)

        assert np.allclose(model_v2.calib_scale.numpy(), payload["calibration"]["scale"])
        assert np.allclose(model_v2.calib_shift.numpy(), payload["calibration"]["shift"])
        # thresholds move with the calibration they were fitted against
        assert np.allclose(model_v2.thresholds.numpy(), payload["thresholds"])
    finally:
        shutil.rmtree(tmp)


def test_tta_path_is_calibrated(model_v2, dummy_input):
    model_v2.eval()
    before = model_v2.predict_with_tta(dummy_input, n_augments=3)["probabilities"]
    model_v2.set_calibration(
        torch.ones(model_v2.num_classes), torch.full((model_v2.num_classes,), -2.5)
    )
    after = model_v2.predict_with_tta(dummy_input, n_augments=3)["probabilities"]
    assert (after < before).all()
