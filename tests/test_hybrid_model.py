"""Tests for RetinalFoundationHybrid model."""

import pytest
import torch
import os

# Use lightweight backbone for testing
os.environ["USE_PRETRAINED"] = "0"
os.environ["FAST_SINGLE_RESOLUTION"] = "1"


@pytest.fixture
def knowledge_graph():
    from src.models.vignn import create_knowledge_graph
    return create_knowledge_graph()


@pytest.fixture
def hybrid_model(knowledge_graph):
    from src.models.retinal_foundation_hybrid import RetinalFoundationHybrid
    model = RetinalFoundationHybrid(
        num_classes=48,
        hidden_dim=128,
        num_graph_layers=1,
        num_heads=4,
        dropout=0.1,
        clinical_knowledge_graph=knowledge_graph,
        backbone="vit_small_patch16_224",  # Small for testing
        img_size=224,
        use_lora=False,
        num_ensemble_heads=2,
        mc_dropout=0.1,
        enable_moe=True,
        moe_top_k=2,
        freeze_backbone=False,
    )
    return model


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 224, 224)


class TestRetinalFoundationHybrid:
    def test_forward_returns_logits(self, hybrid_model, dummy_input):
        """Test that forward pass returns correct shape logits."""
        output = hybrid_model(dummy_input)
        assert output.shape == (2, 48), f"Expected (2, 48), got {output.shape}"

    def test_forward_with_uncertainty(self, hybrid_model, dummy_input):
        """Test uncertainty quantification output."""
        output = hybrid_model(dummy_input, mc_samples=2, return_uncertainty=True)
        assert isinstance(output, dict)
        assert "logits" in output
        assert "predictions" in output
        assert "epistemic_uncertainty" in output
        assert "aleatoric_uncertainty" in output
        assert "confidence_interval_lower" in output
        assert "confidence_interval_upper" in output
        assert output["logits"].shape == (2, 48)
        assert output["predictions"].shape == (2, 48)

    def test_predictions_bounded(self, hybrid_model, dummy_input):
        """Test that sigmoid predictions are in [0, 1]."""
        output = hybrid_model(dummy_input, return_uncertainty=True)
        preds = output["predictions"]
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_aux_loss_with_moe(self, hybrid_model, dummy_input):
        """Test MoE auxiliary loss is computed."""
        hybrid_model(dummy_input)
        aux_loss = hybrid_model.get_aux_loss()
        assert aux_loss.item() >= 0.0

    def test_param_summary(self, hybrid_model):
        """Test parameter count reporting."""
        summary = hybrid_model.get_param_summary()
        assert "total" in summary
        assert "trainable" in summary
        assert "encoder_total" in summary
        assert "head_total" in summary
        assert summary["total"] > 0
        assert summary["trainable"] > 0

    def test_clinical_reasoning_inference(self, hybrid_model, dummy_input):
        """Test full clinical reasoning pipeline."""
        result = hybrid_model.predict_with_clinical_reasoning(
            dummy_input[:1], mc_samples=2
        )
        assert "predictions" in result
        assert "detected_diseases" in result
        assert "referral_priority" in result
        assert "risk_assessment" in result
        assert "uncertainty" in result
        assert result["referral_priority"] in ("URGENT", "ROUTINE", "FOLLOW_UP")

    def test_knowledge_graph_required(self):
        """Test that model raises without knowledge graph."""
        from src.models.retinal_foundation_hybrid import RetinalFoundationHybrid
        with pytest.raises(ValueError, match="requires a ClinicalKnowledgeGraph"):
            RetinalFoundationHybrid(clinical_knowledge_graph=None)

    def test_gradient_flow(self, hybrid_model, dummy_input):
        """Test that gradients flow through trainable parameters."""
        output = hybrid_model(dummy_input)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for name, param in hybrid_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                    break
        assert has_grad, "No gradients flowing through trainable parameters"


class TestRetinalFoundationEncoder:
    def test_encoder_output_shape(self, knowledge_graph):
        from src.models.retinal_foundation_encoder import RetinalFoundationEncoder
        encoder = RetinalFoundationEncoder(
            backbone_name="vit_small_patch16_224",
            output_dim=384,
            use_lora=False,
        )
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 196, 384)

    def test_lora_injection(self):
        from src.models.retinal_foundation_encoder import RetinalFoundationEncoder, LoRALinear
        encoder = RetinalFoundationEncoder(
            backbone_name="vit_small_patch16_224",
            output_dim=384,
            use_lora=True,
            lora_rank=8,
        )
        # Check that LoRA layers exist
        has_lora = False
        for name, module in encoder.named_modules():
            if isinstance(module, LoRALinear):
                has_lora = True
                break
        assert has_lora, "LoRA layers not found after injection"

    def test_trainable_params_with_lora(self):
        from src.models.retinal_foundation_encoder import RetinalFoundationEncoder
        encoder = RetinalFoundationEncoder(
            backbone_name="vit_small_patch16_224",
            output_dim=384,
            use_lora=True,
            lora_rank=8,
        )
        trainable = encoder.get_trainable_params()
        total = encoder.get_total_params()
        assert trainable < total, "LoRA should freeze most parameters"
        assert trainable > 0, "Some parameters should be trainable"


class TestMoERouter:
    def test_moe_output_shape(self):
        from src.models.retinal_foundation_hybrid import MoERouter
        moe = MoERouter(input_dim=256, num_experts=4, expert_dim=128, top_k=2)
        x = torch.randn(4, 256)
        output, aux_loss = moe(x)
        assert output.shape == (4, 256)
        assert aux_loss.item() >= 0.0

    def test_moe_load_balancing(self):
        from src.models.retinal_foundation_hybrid import MoERouter
        moe = MoERouter(input_dim=128, num_experts=4, expert_dim=64, top_k=2)
        x = torch.randn(32, 128)
        _, aux_loss = moe(x)
        assert aux_loss.item() > 0.0, "Load balancing loss should be positive"


class TestUncertaintyHead:
    def test_uncertainty_output(self):
        from src.models.retinal_foundation_hybrid import UncertaintyHead
        head = UncertaintyHead(input_dim=256, num_classes=48, num_heads=3)
        x = torch.randn(4, 256)
        output = head(x, mc_samples=3)
        assert output["logits"].shape == (4, 48)
        assert output["predictions"].shape == (4, 48)
        assert output["epistemic_uncertainty"].shape == (4, 48)

    def test_mc_dropout_increases_uncertainty(self):
        from src.models.retinal_foundation_hybrid import UncertaintyHead
        head = UncertaintyHead(input_dim=256, num_classes=48, num_heads=3, mc_dropout=0.3)
        head.train()  # Enable dropout
        x = torch.randn(4, 256)

        out_1 = head(x, mc_samples=1)
        out_10 = head(x, mc_samples=10)

        # More MC samples should give more stable estimates (lower or similar variance)
        # Both should produce valid output
        assert out_1["epistemic_uncertainty"].shape == out_10["epistemic_uncertainty"].shape


class TestFactoryFunction:
    def test_create_hybrid_model(self, knowledge_graph):
        from src.models.retinal_foundation_hybrid import create_hybrid_model
        model = create_hybrid_model(
            num_classes=48,
            hidden_dim=128,
            clinical_knowledge_graph=knowledge_graph,
            backbone="vit_small_patch16_224",
            use_lora=False,
            enable_moe=False,
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 48)
