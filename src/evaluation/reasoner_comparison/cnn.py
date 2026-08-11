"""``TriageCNN`` — a small convolutional reasoner candidate.

This is the "CNN instead of Qwen" candidate: a MobileNetV3-Small backbone with
three lightweight heads that emit the *structured triage decision* directly from
the image —

* ``priority``      — 4-way softmax over EMERGENCY/URGENT/ROUTINE/FOLLOW_UP,
* ``should_explain``— binary,
* ``should_review`` — binary.

It deliberately does **not** produce a free-text narrative (a CNN can't); in
deployment it would pair with the existing template narrative, which is why the
report scores it as structured-triage-only. MobileNetV3-Small is chosen to match
the operator set of the existing on-device fundus gate / mobile student so the
ONNX export story is already solved.

The trainer here is small on purpose: it backs both the offline smoke micro-train
(a few epochs on CPU over synthetic cases) and the real run (full epochs on
RFMiD with teacher-trace labels). The expensive real run is gated behind the
``GO`` decision in the feasibility doc.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import PRIORITIES, PRIORITY_INDEX, Case

logger = logging.getLogger(__name__)

NUM_PRIORITIES = len(PRIORITIES)


class TriageCNN(nn.Module):
    """MobileNetV3-Small backbone + three triage heads."""

    def __init__(self, backbone: str = "mobilenetv3_small_100", pretrained: bool = False):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 64, 64)).shape[-1]
        self.priority_head = nn.Linear(feat_dim, NUM_PRIORITIES)
        self.explain_head = nn.Linear(feat_dim, 1)
        self.review_head = nn.Linear(feat_dim, 1)
        self.backbone_name = backbone
        logger.info(
            "TriageCNN initialized: backbone=%s feat_dim=%d params=%.2fM",
            backbone,
            feat_dim,
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return (
            self.priority_head(feats),
            self.explain_head(feats).squeeze(-1),
            self.review_head(feats).squeeze(-1),
        )

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6


def _to_image_tensor(image, img_size: int) -> torch.Tensor:
    """Normalize a case image (Tensor or path) to a [3,img_size,img_size] tensor."""
    if isinstance(image, torch.Tensor):
        t = image
    else:  # path -> load + ImageNet-ish scaling
        from PIL import Image
        from torchvision import transforms

        tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        return tf(Image.open(image).convert("RGB"))
    if t.dim() == 3 and t.shape[-1] != img_size:
        t = F.interpolate(
            t.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False
        )[0]
    return t


def _batch(cases: Sequence[Case], img_size: int, device: torch.device) -> torch.Tensor:
    return torch.stack([_to_image_tensor(c.image, img_size) for c in cases]).to(device)


def _labels(cases: Sequence[Case], device: torch.device):
    if any(c.reference is None for c in cases):
        raise ValueError("every training case needs a reference (teacher) label")
    y_prio = torch.tensor([PRIORITY_INDEX[c.reference.priority] for c in cases], device=device)
    y_exp = torch.tensor([float(c.reference.should_explain) for c in cases], device=device)
    y_rev = torch.tensor([float(c.reference.should_review) for c in cases], device=device)
    return y_prio, y_exp, y_rev


def train_triage_cnn(
    train_cases: Sequence[Case],
    *,
    epochs: int = 6,
    lr: float = 1e-3,
    batch_size: int = 32,
    img_size: int = 64,
    device: str | torch.device = "cpu",
    pretrained: bool = False,
    seed: int = 42,
) -> TriageCNN:
    """Train ``TriageCNN`` to imitate the teacher's triage decisions."""
    torch.manual_seed(seed)
    device = torch.device(device)
    model = TriageCNN(pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Class-balanced priority weights so rare EMERGENCY isn't ignored.
    counts = torch.zeros(NUM_PRIORITIES)
    for c in train_cases:
        counts[PRIORITY_INDEX[c.reference.priority]] += 1
    weights = (counts.sum() / counts.clamp(min=1)).to(device)
    weights[counts == 0] = 0.0

    n = len(train_cases)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            batch = [train_cases[i] for i in perm[start : start + batch_size]]
            x = _batch(batch, img_size, device)
            y_prio, y_exp, y_rev = _labels(batch, device)
            opt.zero_grad()
            logit_prio, logit_exp, logit_rev = model(x)
            loss = (
                F.cross_entropy(logit_prio, y_prio, weight=weights)
                + F.binary_cross_entropy_with_logits(logit_exp, y_exp)
                + F.binary_cross_entropy_with_logits(logit_rev, y_rev)
            )
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(batch)
        logger.info("TriageCNN epoch %d/%d loss=%.4f", epoch + 1, epochs, total / n)

    model.eval()
    return model
