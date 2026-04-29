"""Model Card generator following Google's Model Cards framework (2024).
Generates standardized documentation for each model version."""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelDetails:
    name: str
    version: str
    type: str  # e.g., "Graph Neural Network"
    architecture: str
    num_parameters: int = 0
    training_date: str = ""
    framework: str = "PyTorch 2.0+"
    license: str = "CC-BY-4.0"
    contact: str = ""
    citation: str = ""


@dataclass
class IntendedUse:
    primary_use: str = "Screening aid for multi-label retinal disease detection from fundus images"
    primary_users: list[str] = field(default_factory=lambda: ["Ophthalmologists", "Clinical researchers", "Healthcare AI teams"])
    out_of_scope: list[str] = field(default_factory=lambda: [
        "Standalone diagnostic tool without clinician oversight",
        "Pediatric retinal screening (model trained on adult population)",
        "Non-fundus imaging modalities (OCT, fluorescein angiography)",
        "Real-time surgical guidance",
    ])


@dataclass
class TrainingData:
    dataset: str = "RFMiD (Retinal Fundus Multi-Disease Image Dataset)"
    source: str = "Kaggle: mpairwelauben/multi-disease-retinal-eye-disease-dataset"
    num_samples: int = 0
    num_classes: int = 45
    split_ratios: str = "70/20/10 (train/val/test)"
    preprocessing: str = "Resize to 224x224, ImageNet normalization"
    augmentation: str = "Random flips, rotation(15°), color jitter, random erasing"
    known_limitations: list[str] = field(default_factory=lambda: [
        "Dataset may not represent all global demographics equally",
        "Severe class imbalance — rare diseases have very few samples",
        "Image quality varies across collection sites",
    ])


@dataclass
class PerformanceMetrics:
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    auc_roc: float = 0.0
    mAP: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    hamming_loss: float = 0.0
    inference_ms: float = 0.0
    per_class_metrics: dict = field(default_factory=dict)


@dataclass
class FairnessAnalysis:
    evaluated: bool = False
    methodology: str = ""
    findings: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)


@dataclass
class EthicalConsiderations:
    risks: list[str] = field(default_factory=lambda: [
        "False negatives may delay treatment for sight-threatening conditions",
        "False positives may cause unnecessary patient anxiety and follow-up costs",
        "Model performance may vary across demographic groups not well-represented in training data",
        "Over-reliance on AI screening without clinical validation could harm patients",
    ])
    mitigations: list[str] = field(default_factory=lambda: [
        "Model is designed as a screening AID, not a replacement for clinical diagnosis",
        "High-confidence threshold recommended for automated referral decisions",
        "Human-in-the-loop review required for all positive findings",
        "Regular performance monitoring across demographic subgroups",
        "Clinical validation study recommended before deployment",
    ])


@dataclass
class RegulatoryInfo:
    classification: str = "High-risk AI system (EU AI Act Article 6)"
    applicable_regulations: list[str] = field(default_factory=lambda: [
        "EU AI Act (2024/1689) — High-risk AI system",
        "EU MDR 2017/745 — Software as Medical Device (SaMD)",
        "FDA 21 CFR Part 820 — Quality System Regulation (if US market)",
        "ISO 13485 — Medical Device Quality Management",
        "ISO 14971 — Risk Management for Medical Devices",
        "IEC 62304 — Medical Device Software Lifecycle",
    ])
    conformity_status: str = "Pre-conformity — research use only"
    post_market_surveillance: str = "Continuous monitoring via drift detection and prediction logging"


@dataclass
class ModelCard:
    model_details: ModelDetails = field(default_factory=lambda: ModelDetails(name="", version="", type="", architecture=""))
    intended_use: IntendedUse = field(default_factory=IntendedUse)
    training_data: TrainingData = field(default_factory=TrainingData)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    fairness: FairnessAnalysis = field(default_factory=FairnessAnalysis)
    ethical_considerations: EthicalConsiderations = field(default_factory=EthicalConsiderations)
    regulatory: RegulatoryInfo = field(default_factory=RegulatoryInfo)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Model card saved to {path}")

    def to_markdown(self, path: Path):
        """Generate human-readable model card markdown."""
        md = self._render_markdown()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md)
        logger.info(f"Model card markdown saved to {path}")

    def _render_markdown(self) -> str:
        d = self.model_details
        lines = [
            f"# Model Card: {d.name} v{d.version}",
            f"*Generated: {self.generated_at}*\n",
            "## Model Details",
            f"- **Architecture:** {d.architecture}",
            f"- **Type:** {d.type}",
            f"- **Parameters:** {d.num_parameters:,}",
            f"- **Framework:** {d.framework}",
            f"- **License:** {d.license}\n",
            "## Intended Use",
            f"**Primary:** {self.intended_use.primary_use}\n",
            "**Users:** " + ", ".join(self.intended_use.primary_users) + "\n",
            "**Out of Scope:**",
        ]
        for item in self.intended_use.out_of_scope:
            lines.append(f"- {item}")

        lines += ["\n## Training Data", f"- **Dataset:** {self.training_data.dataset}",
                   f"- **Samples:** {self.training_data.num_samples}", f"- **Classes:** {self.training_data.num_classes}",
                   f"- **Split:** {self.training_data.split_ratios}\n",
                   "**Known Limitations:**"]
        for item in self.training_data.known_limitations:
            lines.append(f"- {item}")

        p = self.performance
        lines += ["\n## Performance", "| Metric | Value |", "|--------|-------|",
                   f"| F1 Macro | {p.f1_macro:.4f} |", f"| F1 Micro | {p.f1_micro:.4f} |",
                   f"| AUC-ROC | {p.auc_roc:.4f} |", f"| mAP | {p.mAP:.4f} |",
                   f"| Precision | {p.precision_macro:.4f} |", f"| Recall | {p.recall_macro:.4f} |",
                   f"| Inference | {p.inference_ms:.1f}ms |"]

        lines += ["\n## Ethical Considerations", "### Risks"]
        for r in self.ethical_considerations.risks:
            lines.append(f"- {r}")
        lines.append("\n### Mitigations")
        for m in self.ethical_considerations.mitigations:
            lines.append(f"- {m}")

        lines += ["\n## Regulatory", f"- **Classification:** {self.regulatory.classification}",
                   f"- **Status:** {self.regulatory.conformity_status}\n", "**Applicable Regulations:**"]
        for r in self.regulatory.applicable_regulations:
            lines.append(f"- {r}")

        return "\n".join(lines)


def generate_model_card(
    model_name: str,
    version: str,
    architecture: str,
    num_parameters: int,
    metrics: dict,
    num_samples: int = 0,
) -> ModelCard:
    """Factory to create a pre-filled model card."""
    card = ModelCard()
    card.model_details = ModelDetails(
        name=model_name, version=version, type="Graph Neural Network",
        architecture=architecture, num_parameters=num_parameters,
        training_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    card.training_data.num_samples = num_samples
    card.performance = PerformanceMetrics(
        f1_macro=metrics.get("f1_macro", 0), f1_micro=metrics.get("f1_micro", 0),
        auc_roc=metrics.get("auc_roc", 0), mAP=metrics.get("mAP", 0),
        precision_macro=metrics.get("precision_macro", 0), recall_macro=metrics.get("recall_macro", 0),
        hamming_loss=metrics.get("hamming_loss", 0), inference_ms=metrics.get("inference_ms", 0),
        per_class_metrics=metrics.get("per_class", {}),
    )
    return card
