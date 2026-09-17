"""Tests for DoRA (Weight-Decomposed LoRA) and Speculative Vision Gating."""

import torch
import torch.nn as nn

from src.models.dora import DoRALinear, SpeculativeEarlyExitGate, convert_linear_to_dora


def test_dora_linear_initialization_and_forward():
    in_dim, out_dim = 64, 128
    original = nn.Linear(in_dim, out_dim)

    # Initial forward on original layer
    x = torch.randn(4, in_dim)
    orig_out = original(x)

    dora = DoRALinear(original, rank=8, alpha=16.0)

    # At init, lora_B is zeros, so DoRA output must match original output exactly!
    dora_out = dora(x)
    assert torch.allclose(orig_out, dora_out, atol=1e-5)


def test_dora_trainable_parameters():
    original = nn.Linear(32, 64)
    dora = DoRALinear(original, rank=4)

    # Original weights must be frozen
    assert not original.weight.requires_grad
    if original.bias is not None:
        assert not original.bias.requires_grad

    # Magnitude and LoRA A, B must be trainable
    assert dora.magnitude.requires_grad
    assert dora.lora_A.weight.requires_grad
    assert dora.lora_B.weight.requires_grad


def test_dora_merge_weights():
    original = nn.Linear(32, 64)
    dora = DoRALinear(original, rank=4)
    merged = dora.merge_weights()

    assert merged.shape == (64, 32)
    # Check that merged forward equals dora forward
    x = torch.randn(2, 32)
    dora_out = dora(x)
    merged_out = torch.nn.functional.linear(x, merged, original.bias)
    assert torch.allclose(dora_out, merged_out, atol=1e-5)


def test_convert_linear_to_dora_recursive():
    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(32, 96)
            self.proj = nn.Linear(96, 32)
            self.fc = nn.Linear(32, 10)

    model = ToyBlock()
    converted = convert_linear_to_dora(model, target_layer_names=("qkv", "proj"), rank=4)
    assert converted == 2
    assert isinstance(model.qkv, DoRALinear)
    assert isinstance(model.proj, DoRALinear)
    assert isinstance(model.fc, nn.Linear)  # fc untouched


def test_speculative_early_exit_gate():
    gate = SpeculativeEarlyExitGate(embed_dim=128, confidence_threshold=0.95)
    cls_token = torch.randn(4, 128)

    prob_normal, should_exit = gate(cls_token)
    assert prob_normal.shape == (4, 1)
    assert should_exit.shape == (4,)
    assert should_exit.dtype == torch.bool
    assert (prob_normal >= 0.0).all() and (prob_normal <= 1.0).all()
