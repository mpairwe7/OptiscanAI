"""Case sources for the harness: a self-contained synthetic generator (smoke
mode) and a real RFMiD loader (gated full run).

The synthetic generator needs no GPU, no data, and no API keys, so the harness
and its tests run end-to-end on CPU in seconds. Each synthetic case gets:

* a random disease-probability profile over a small clinical vocabulary,
* the resulting detected ``predictions`` (prob >= threshold),
* a **synthetic teacher** ``reference`` (rules + a little nuance) used as ground
  truth for scoring and as the CNN's training target, and
* a synthetic ``image`` tensor that *encodes* the probability profile, so a CNN
  has a learnable, label-correlated signal (this is a plumbing/learnability
  fixture, not real fundus data — see ``docs/28-reasoner-cnn-vs-distilledqwen.md``).

The real loader mirrors this shape but sources predictions from
``model_service`` over RFMiD images and ``reference`` from a teacher-trace JSONL
(see ``scripts/generate_reasoning_traces.py``). It is intentionally lazy and is
not exercised by the smoke path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .interface import (
    CRITICAL_CODES,
    EMERGENCY_CODES,
    Case,
    Prediction,
    ReasonerOutput,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Small clinical vocabulary spanning the production tiers. Codes/names are drawn
# from the RFMiD / V2 disease set plus the emergency codes that triage_node keys
# on (CRAO/AION), so the EMERGENCY path is exercised.
DISEASE_VOCAB: list[tuple[str, str]] = [
    ("CRAO", "Central Retinal Artery Occlusion"),
    ("AION", "Anterior Ischemic Optic Neuropathy"),
    ("CRVO", "Central Retinal Vein Occlusion"),
    ("VH", "Vitreous Hemorrhage"),
    ("RS", "Retinoschisis"),
    ("DR", "Diabetic Retinopathy"),
    ("ARMD", "Age-Related Macular Degeneration"),
    ("BRVO", "Branch Retinal Vein Occlusion"),
    ("CSR", "Central Serous Retinopathy"),
    ("MH", "Macular Hole"),
    ("ERM", "Epiretinal Membrane"),
    ("DN", "Drusen"),
    ("MYA", "Myopia"),
    ("ODC", "Optic Disc Cupping"),
    ("TSLN", "Tessellation"),
    ("LS", "Laser Scars"),
]
CODE_TO_NAME: dict[str, str] = dict(DISEASE_VOCAB)
NUM_DISEASES = len(DISEASE_VOCAB)


def synthetic_teacher(predictions: list[Prediction], referral: str) -> ReasonerOutput:
    """A deterministic stand-in for the teacher LLM over a single case.

    Starts from the production rule logic (``src/agents/graph.py:triage_node``)
    and adds two nuance adjustments a strong LLM would plausibly make, so the
    target is *learnable but not identical to the naive rules* — that gap is what
    lets the metrics tell candidates apart.
    """
    codes = [p.code for p in predictions]
    has_emergency = any(c in EMERGENCY_CODES for c in codes)
    has_critical = any(c in CRITICAL_CODES for c in codes)
    n = len(predictions)
    low_conf = any(p.probability < 0.70 for p in predictions)
    max_prob = max((p.probability for p in predictions), default=0.0)

    # Base priority = rule behaviour.
    priority = "EMERGENCY" if has_emergency else referral
    # Nuance 1: a dense multi-disease presentation with high confidence is at
    # least URGENT even if individual referrals were ROUTINE.
    if priority in ("ROUTINE", "FOLLOW_UP") and n >= 3 and max_prob >= 0.85:
        priority = "URGENT"
    # Nuance 2: a single, low-confidence, non-critical finding de-escalates to
    # FOLLOW_UP (avoid over-referral).
    if priority == "ROUTINE" and n == 1 and low_conf and not has_critical:
        priority = "FOLLOW_UP"

    should_explain = has_critical or n >= 3
    should_review = low_conf or n > 5 or has_emergency

    if has_emergency:
        reasoning = "Sight-threatening occlusion detected — immediate referral required"
    elif has_critical:
        reasoning = f"Critical pathology among {n} findings — specialist review recommended"
    elif n > 5:
        reasoning = f"Complex multi-disease presentation ({n} findings) — review for co-management"
    elif n > 0:
        reasoning = f"{n} finding(s) detected at {priority} priority"
    else:
        reasoning = "No significant pathology detected"

    narrative = _reference_narrative(predictions, priority)
    return ReasonerOutput(
        priority=priority,
        should_explain=should_explain,
        should_review=should_review,
        reasoning=reasoning,
        narrative=narrative,
        source="synthetic_teacher",
    )


def _reference_narrative(predictions: list[Prediction], priority: str) -> str:
    if not predictions:
        return (
            "AI screening found no significant retinal pathology. "
            "Routine follow-up recommended. This is an AI-assisted screening result."
        )
    top = predictions[0]
    listed = ", ".join(f"{p.name} ({p.code}) at {p.probability:.0%}" for p in predictions[:4])
    return (
        f"AI screening identified {len(predictions)} finding(s): {listed}. "
        f"Primary finding is {top.name} ({top.code}). Referral priority: {priority}. "
        "Specialist confirmation is required for this AI-assisted result."
    )


def make_synthetic_cases(
    n: int = 240,
    seed: int = 42,
    threshold: float = 0.5,
    img_size: int = 64,
    with_images: bool = True,
) -> list[Case]:
    """Generate ``n`` synthetic cases with teacher references and encoded images."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cases: list[Case] = []

    for i in range(n):
        # Sparse-ish probability profile: most diseases low, a few elevated.
        base = rng.beta(0.4, 6.0, size=NUM_DISEASES)  # mostly small
        n_active = int(rng.integers(0, 6))
        active_idx = rng.choice(NUM_DISEASES, size=n_active, replace=False)
        base[active_idx] = rng.uniform(0.55, 0.98, size=n_active)
        probs = {DISEASE_VOCAB[j][0]: float(base[j]) for j in range(NUM_DISEASES)}

        predictions = [
            Prediction(
                code=code,
                name=CODE_TO_NAME[code],
                probability=probs[code],
                confidence="high" if probs[code] >= 0.8 else "medium",
            )
            for code in probs
            if probs[code] >= threshold
        ]
        predictions.sort(key=lambda p: p.probability, reverse=True)

        referral = _referral_from_codes([p.code for p in predictions])
        ref = synthetic_teacher(predictions, referral)

        image = _encode_image(base, img_size, rng) if with_images else None
        cases.append(
            Case(
                scan_id=f"synthetic-{i:04d}",
                predictions=predictions,
                probabilities=probs,
                referral_priority=referral,
                image=image,
                reference=ref,
                metadata={"synthetic": True},
            )
        )

    return cases


def _referral_from_codes(codes: list[str]) -> str:
    """Mirror ClinicalKnowledgeGraph.get_referral_priority (vignn.py:428)."""
    if any(c in CRITICAL_CODES for c in codes):
        return "URGENT"
    if codes:
        return "ROUTINE"
    return "FOLLOW_UP"


def _encode_image(profile, img_size: int, rng) -> "torch.Tensor":
    """Paint the disease-probability profile into a small RGB tensor.

    Each disease maps to a grid cell whose intensity is its probability; the CNN
    learns the rule mapping from this spatial encoding. Light per-channel jitter
    + Gaussian noise prevent a degenerate identity solution.
    """
    import math

    import numpy as np
    import torch

    grid = math.ceil(math.sqrt(NUM_DISEASES))
    canvas = np.zeros((grid, grid), dtype=np.float32)
    for idx, val in enumerate(profile):
        canvas[idx // grid, idx % grid] = val
    img = np.repeat(
        np.repeat(canvas, math.ceil(img_size / grid), axis=0), math.ceil(img_size / grid), axis=1
    )
    img = img[:img_size, :img_size]
    rgb = np.stack([img * 1.0, img * 0.9, img * 1.1], axis=0)
    rgb = rgb + rng.normal(0, 0.02, size=rgb.shape).astype(np.float32)
    return torch.from_numpy(np.clip(rgb, 0.0, 1.0))


def load_real_cases(
    images_dir: str | Path,
    traces_path: str | Path,
    limit: int | None = None,
) -> list[Case]:
    """Load real RFMiD cases: predictions from ``model_service``, references from
    a teacher-trace JSONL. Lazy + gated — not used by the smoke path.

    The trace JSONL is one object per line with at least ``scan_id`` and the
    reference triage fields (see ``scripts/generate_reasoning_traces.py``).
    """
    import json

    from PIL import Image

    from backend.app.core.model_service import model_service

    traces: dict[str, dict] = {}
    with open(traces_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                traces[obj["scan_id"]] = obj

    images = sorted(Path(images_dir).glob("*.png")) + sorted(Path(images_dir).glob("*.jpg"))
    if limit:
        images = images[:limit]

    cases: list[Case] = []
    for img_path in images:
        scan_id = img_path.stem
        trace = traces.get(scan_id)
        if trace is None:
            logger.warning("no teacher trace for %s — skipping", scan_id)
            continue
        result = model_service.predict(Image.open(img_path).convert("RGB"))
        predictions = [
            Prediction(
                code=p["code"],
                name=p["name"],
                probability=float(p["probability"]),
                confidence=p.get("confidence", "unknown"),
            )
            for p in result.get("predictions", [])
        ]
        probs = {
            k: (v["probability"] if isinstance(v, dict) else float(v))
            for k, v in result.get("all_probabilities", {}).items()
        }
        ref = ReasonerOutput(
            priority=trace["priority"],
            should_explain=bool(trace.get("should_explain", False)),
            should_review=bool(trace.get("should_review", False)),
            reasoning=trace.get("reasoning", ""),
            narrative=trace.get("narrative", ""),
            source="teacher",
        )
        cases.append(
            Case(
                scan_id=scan_id,
                predictions=predictions,
                probabilities=probs,
                referral_priority=result.get("clinical", {}).get("referral_priority", "FOLLOW_UP"),
                image=img_path,
                reference=ref,
            )
        )
    return cases
