"""Tests for quantization and optimization pipeline."""

import pytest
import torch
import torch.nn as nn
import os

os.environ["USE_PRETRAINED"] = "0"


class SimpleModel(nn.Module):
    """Simple model for testing quantization."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 48)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def simple_model():
    return SimpleModel()


class TestDynamicQuantization:
    def test_dynamic_int8_reduces_size(self, simple_model):
        from src.optimization.quantization import quantize_dynamic_int8, _model_size_mb
        orig_size = _model_size_mb(simple_model)
        quantized = quantize_dynamic_int8(simple_model)
        quant_size = _model_size_mb(quantized)
        # INT8 should be smaller (at least for Linear layers)
        assert quant_size <= orig_size

    def test_dynamic_int8_output_shape(self, simple_model):
        from src.optimization.quantization import quantize_dynamic_int8
        quantized = quantize_dynamic_int8(simple_model)
        x = torch.randn(4, 256)
        output = quantized(x)
        assert output.shape == (4, 48)


class TestFP16Conversion:
    def test_fp16_conversion(self, simple_model):
        from src.optimization.quantization import convert_to_fp16
        fp16_model = convert_to_fp16(simple_model)
        # Check that parameters are float16
        for p in fp16_model.parameters():
            assert p.dtype == torch.float16


class TestDistillationLoss:
    def test_distillation_loss_computes(self):
        from src.optimization.quantization import DistillationLoss
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.7)
        student_logits = torch.randn(4, 48)
        teacher_logits = torch.randn(4, 48)
        targets = torch.zeros(4, 48)
        loss = loss_fn(student_logits, teacher_logits, targets)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_distillation_loss_gradient(self):
        from src.optimization.quantization import DistillationLoss
        loss_fn = DistillationLoss()
        student_logits = torch.randn(4, 48, requires_grad=True)
        teacher_logits = torch.randn(4, 48)
        targets = torch.zeros(4, 48)
        loss = loss_fn(student_logits, teacher_logits, targets)
        loss.backward()
        assert student_logits.grad is not None


class TestBenchmark:
    def test_benchmark_cpu(self, simple_model):
        from src.optimization.quantization import benchmark_latency
        result = benchmark_latency(
            simple_model, input_shape=(1, 256), device="cpu",
            warmup_runs=5, benchmark_runs=10,
        )
        assert "mean_ms" in result
        assert "p99_ms" in result
        assert "throughput_fps" in result
        assert result["mean_ms"] > 0


class TestModelSize:
    def test_model_size_mb(self, simple_model):
        from src.optimization.quantization import _model_size_mb
        size = _model_size_mb(simple_model)
        assert size > 0
