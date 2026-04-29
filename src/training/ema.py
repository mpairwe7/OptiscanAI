"""
Exponential Moving Average (EMA) for model weights.
Maintains a shadow copy that tracks a decayed average of parameters,
typically yielding 1-2% improvement on validation metrics.
"""
from __future__ import annotations
import copy
from collections import OrderedDict

import torch
import torch.nn as nn


class ModelEMA:
    """
    EMA of model parameters.
    Call .update() after each optimizer step.
    Use .state_dict() for checkpointing and .apply() for evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow params: shadow = decay * shadow + (1-decay) * model."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def state_dict(self) -> OrderedDict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: OrderedDict):
        self.shadow.load_state_dict(state_dict)

    def apply(self, model: nn.Module):
        """Copy EMA weights into model (for evaluation)."""
        model.load_state_dict(self.shadow.state_dict())

    def module(self) -> nn.Module:
        """Return the shadow model directly."""
        return self.shadow
