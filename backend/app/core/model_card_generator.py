"""Auto-generate model cards after MLflow model promotion.

Listens for ``EventType.MODEL_PROMOTED`` on the event bus and produces
a standardised model card (JSON + Markdown) in the configured output
directory.  Re-uses the governance-layer ``ModelCard`` dataclass when
available, with a fallback dict-based generator for environments where
the governance package is not on ``sys.path``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root -- needed so ``src.*`` imports resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional re-use of the governance-layer ModelCard
# ---------------------------------------------------------------------------
try:
    from src.governance.model_card import (
        FairnessAnalysis,
        ModelDetails,
        PerformanceMetrics,
    )
    from src.governance.model_card import (
        ModelCard as _GovernanceModelCard,
    )
    from src.governance.model_card import (
        generate_model_card as _gov_generate,
    )

    _GOV_AVAILABLE = True
except ImportError:
    _GovernanceModelCard = None  # type: ignore[assignment,misc]
    _GOV_AVAILABLE = False
    logger.info(
        "src.governance.model_card not importable -- "
        "model card generator will use built-in templates"
    )


class ModelCardAutoGenerator:
    """Generate, save, and render model cards triggered by MLflow promotion.

    Configuration is read from ``settings.model_card``:
        - ``auto_generate`` (bool): master switch
        - ``output_dir`` (str): where JSON/Markdown artefacts are written
    """

    def __init__(self) -> None:
        self._cfg = settings.model_card
        self._auto_generate: bool = self._cfg.auto_generate
        self._output_dir = Path(self._cfg.output_dir)
        logger.info(
            "ModelCardAutoGenerator initialised " "(auto_generate=%s, output_dir=%s)",
            self._auto_generate,
            self._output_dir,
        )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model_name: str,
        model_version: str,
        metrics: dict[str, Any],
        training_config: Optional[dict[str, Any]] = None,
        fairness_report: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Build a complete model card dictionary.

        Sections:
            model_details, intended_use, training_data,
            evaluation_results, ethical_considerations,
            limitations, regulatory_status

        If the governance-layer ``ModelCard`` dataclass is available the
        heavy lifting is delegated there; otherwise we build an equivalent
        dict from scratch.

        Parameters
        ----------
        model_name:
            Human-readable model identifier (e.g. ``retinalai-vignn``).
        model_version:
            Semantic or registry version string.
        metrics:
            Evaluation metrics dict -- keys such as ``f1_macro``, ``auc_roc``,
            ``precision_macro``, ``recall_macro``, ``mAP``, ``inference_ms``.
        training_config:
            Optional hyperparameter / dataset metadata.
        fairness_report:
            Optional fairness evaluation output.

        Returns
        -------
        dict
            The full model card as a plain dictionary.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Try the governance-layer generator first ---
        if _GOV_AVAILABLE:
            try:
                architecture = (training_config or {}).get(
                    "architecture", "ViGNN (Vision Graph Neural Network)"
                )
                num_parameters = int((training_config or {}).get("num_parameters", 0))
                num_samples = int((training_config or {}).get("num_samples", 0))

                gov_card = _gov_generate(
                    model_name=model_name,
                    version=model_version,
                    architecture=architecture,
                    num_parameters=num_parameters,
                    metrics=metrics,
                    num_samples=num_samples,
                )

                # Overlay fairness if supplied
                if fairness_report:
                    gov_card.fairness = FairnessAnalysis(
                        evaluated=True,
                        methodology=fairness_report.get("methodology", ""),
                        findings=fairness_report.get("findings", []),
                        mitigations=fairness_report.get("mitigations", []),
                    )

                card_dict = gov_card.to_dict()
                card_dict["generated_at"] = now_iso
                return card_dict
            except Exception:
                logger.warning(
                    "Governance ModelCard generation failed -- falling back",
                    exc_info=True,
                )

        # --- Fallback: manual dict construction ---
        card: dict[str, Any] = {
            "model_details": {
                "name": model_name,
                "version": model_version,
                "type": "Graph Neural Network",
                "architecture": (training_config or {}).get(
                    "architecture", "ViGNN (Vision Graph Neural Network)"
                ),
                "num_parameters": int((training_config or {}).get("num_parameters", 0)),
                "framework": "PyTorch 2.0+",
                "license": "CC-BY-4.0",
                "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            },
            "intended_use": {
                "primary_use": (
                    "Screening aid for multi-label retinal disease detection " "from fundus images"
                ),
                "primary_users": [
                    "Ophthalmologists",
                    "Clinical researchers",
                    "Healthcare AI teams",
                ],
                "out_of_scope": [
                    "Standalone diagnostic tool without clinician oversight",
                    "Pediatric retinal screening (model trained on adult population)",
                    "Non-fundus imaging modalities (OCT, fluorescein angiography)",
                    "Real-time surgical guidance",
                ],
            },
            "training_data": {
                "dataset": "RFMiD (Retinal Fundus Multi-Disease Image Dataset)",
                "source": "Kaggle: mpairwelauben/multi-disease-retinal-eye-disease-dataset",
                "num_samples": int((training_config or {}).get("num_samples", 0)),
                "num_classes": 45,
                "split_ratios": "70/20/10 (train/val/test)",
                "preprocessing": "Resize to 224x224, ImageNet normalization",
                "augmentation": ("Random flips, rotation(15 deg), color jitter, random erasing"),
                "known_limitations": [
                    "Dataset may not represent all global demographics equally",
                    "Severe class imbalance -- rare diseases have very few samples",
                    "Image quality varies across collection sites",
                ],
            },
            "evaluation_results": {
                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                "f1_micro": float(metrics.get("f1_micro", 0.0)),
                "auc_roc": float(metrics.get("auc_roc", 0.0)),
                "mAP": float(metrics.get("mAP", 0.0)),
                "precision_macro": float(metrics.get("precision_macro", 0.0)),
                "recall_macro": float(metrics.get("recall_macro", 0.0)),
                "hamming_loss": float(metrics.get("hamming_loss", 0.0)),
                "inference_ms": float(metrics.get("inference_ms", 0.0)),
                "per_class_metrics": metrics.get("per_class", {}),
            },
            "ethical_considerations": {
                "risks": [
                    "False negatives may delay treatment for sight-threatening conditions",
                    "False positives may cause unnecessary patient anxiety and follow-up costs",
                    (
                        "Model performance may vary across demographic groups "
                        "not well-represented in training data"
                    ),
                    (
                        "Over-reliance on AI screening without clinical "
                        "validation could harm patients"
                    ),
                ],
                "mitigations": [
                    "Model is designed as a screening AID, not a replacement for clinical diagnosis",
                    "High-confidence threshold recommended for automated referral decisions",
                    "Human-in-the-loop review required for all positive findings",
                    "Regular performance monitoring across demographic subgroups",
                    "Clinical validation study recommended before deployment",
                ],
            },
            "limitations": [
                "Not validated for pediatric populations",
                "Performance may degrade on images from unseen camera devices",
                "Rare diseases with <10 training samples have low recall",
                "Requires 224x224 RGB fundus images as input",
            ],
            "regulatory_status": {
                "classification": "High-risk AI system (EU AI Act Article 6)",
                "applicable_regulations": [
                    "EU AI Act (2024/1689) -- High-risk AI system",
                    "EU MDR 2017/745 -- Software as Medical Device (SaMD)",
                    "FDA 21 CFR Part 820 -- Quality System Regulation (if US market)",
                    "ISO 13485 -- Medical Device Quality Management",
                    "ISO 14971 -- Risk Management for Medical Devices",
                    "IEC 62304 -- Medical Device Software Lifecycle",
                ],
                "conformity_status": "Pre-conformity -- research use only",
                "post_market_surveillance": (
                    "Continuous monitoring via drift detection and prediction logging"
                ),
                "regulatory_mode": settings.regulatory_mode,
            },
            "generated_at": now_iso,
        }

        # Merge fairness
        if fairness_report:
            card["fairness_analysis"] = {
                "evaluated": True,
                "methodology": fairness_report.get("methodology", ""),
                "findings": fairness_report.get("findings", []),
                "mitigations": fairness_report.get("mitigations", []),
            }
        else:
            card["fairness_analysis"] = {
                "evaluated": False,
                "methodology": "",
                "findings": [],
                "mitigations": [],
            }

        return card

    # ------------------------------------------------------------------
    # Event bus handler
    # ------------------------------------------------------------------

    async def on_model_promoted(self, event: Any) -> None:
        """Handle ``EventType.MODEL_PROMOTED`` from the event bus.

        Extracts model name, version, and validation metrics from the
        event payload, generates a model card, saves it, and emits a
        ``MODEL_CARD_GENERATED`` event.
        """
        if not self._auto_generate:
            logger.debug("Model card auto-generation disabled -- skipping")
            return

        data = event.data if hasattr(event, "data") else {}
        model_name = data.get("model_name", "retinalai-vignn")
        model_version = str(data.get("model_version", "unknown"))
        validation_metrics = data.get("validation_metrics", {})
        training_config = data.get("training_config")
        fairness_report = data.get("fairness_report")

        logger.info(
            "Generating model card for %s v%s after promotion",
            model_name,
            model_version,
        )

        try:
            card = self.generate(
                model_name=model_name,
                model_version=model_version,
                metrics=validation_metrics,
                training_config=training_config,
                fairness_report=fairness_report,
            )
            self.save_card(card, version=model_version)

            # Emit downstream event
            try:
                from src.agents.event_bus import Event, EventType, event_bus

                await event_bus.emit(
                    Event(
                        type=EventType.MODEL_CARD_GENERATED,
                        source="model_card_generator",
                        data={
                            "model_name": model_name,
                            "model_version": model_version,
                            "output_dir": str(self._output_dir),
                        },
                    )
                )
            except Exception:
                logger.debug(
                    "Could not emit MODEL_CARD_GENERATED event",
                    exc_info=True,
                )

        except Exception:
            logger.exception(
                "Failed to generate model card for %s v%s",
                model_name,
                model_version,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_card(self, card: dict, version: str) -> None:
        """Save the model card as both JSON and Markdown.

        Files are written to ``<output_dir>/model_card_v<version>.json``
        and ``<output_dir>/model_card_v<version>.md``.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        safe_version = version.replace("/", "_").replace("\\", "_")

        json_path = self._output_dir / f"model_card_v{safe_version}.json"
        json_path.write_text(json.dumps(card, indent=2, default=str))
        logger.info("Model card JSON saved: %s", json_path)

        md_path = self._output_dir / f"model_card_v{safe_version}.md"
        md_path.write_text(self.to_markdown(card))
        logger.info("Model card Markdown saved: %s", md_path)

    def to_markdown(self, card: dict) -> str:
        """Render a model card dict as human-readable Markdown."""
        details = card.get("model_details", {})
        intended = card.get("intended_use", {})
        training = card.get("training_data", {})
        evaluation = card.get("evaluation_results", {})
        ethics = card.get("ethical_considerations", {})
        limitations = card.get("limitations", [])
        regulatory = card.get("regulatory_status", {})
        fairness = card.get("fairness_analysis", {})
        generated_at = card.get("generated_at", "")

        lines: list[str] = []

        # Header
        name = details.get("name", "RetinalAI")
        version = details.get("version", "")
        lines.append(f"# Model Card: {name} v{version}")
        lines.append(f"*Generated: {generated_at}*")
        lines.append("")

        # Model Details
        lines.append("## Model Details")
        lines.append(f"- **Architecture:** {details.get('architecture', 'N/A')}")
        lines.append(f"- **Type:** {details.get('type', 'N/A')}")
        num_params = details.get("num_parameters", 0)
        lines.append(
            f"- **Parameters:** {num_params:,}"
            if isinstance(num_params, int)
            else f"- **Parameters:** {num_params}"
        )
        lines.append(f"- **Framework:** {details.get('framework', 'PyTorch 2.0+')}")
        lines.append(f"- **License:** {details.get('license', 'CC-BY-4.0')}")
        lines.append(f"- **Training Date:** {details.get('training_date', 'N/A')}")
        lines.append("")

        # Intended Use
        lines.append("## Intended Use")
        lines.append(f"**Primary:** {intended.get('primary_use', 'N/A')}")
        lines.append("")
        users = intended.get("primary_users", [])
        if users:
            lines.append("**Users:** " + ", ".join(users))
            lines.append("")
        out_of_scope = intended.get("out_of_scope", [])
        if out_of_scope:
            lines.append("**Out of Scope:**")
            for item in out_of_scope:
                lines.append(f"- {item}")
            lines.append("")

        # Training Data
        lines.append("## Training Data")
        lines.append(f"- **Dataset:** {training.get('dataset', 'N/A')}")
        lines.append(f"- **Samples:** {training.get('num_samples', 'N/A')}")
        lines.append(f"- **Classes:** {training.get('num_classes', 45)}")
        lines.append(f"- **Split:** {training.get('split_ratios', 'N/A')}")
        lines.append(f"- **Preprocessing:** {training.get('preprocessing', 'N/A')}")
        lines.append("")
        known_lim = training.get("known_limitations", [])
        if known_lim:
            lines.append("**Known Data Limitations:**")
            for item in known_lim:
                lines.append(f"- {item}")
            lines.append("")

        # Evaluation Results
        lines.append("## Evaluation Results")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        metric_keys = [
            ("f1_macro", "F1 Macro"),
            ("f1_micro", "F1 Micro"),
            ("auc_roc", "AUC-ROC"),
            ("mAP", "mAP"),
            ("precision_macro", "Precision (Macro)"),
            ("recall_macro", "Recall (Macro)"),
            ("hamming_loss", "Hamming Loss"),
            ("inference_ms", "Inference (ms)"),
        ]
        for key, label in metric_keys:
            val = evaluation.get(key, 0.0)
            if key == "inference_ms":
                lines.append(f"| {label} | {val:.1f} |")
            else:
                lines.append(f"| {label} | {val:.4f} |")
        lines.append("")

        # Ethical Considerations
        lines.append("## Ethical Considerations")
        risks = ethics.get("risks", [])
        if risks:
            lines.append("### Risks")
            for r in risks:
                lines.append(f"- {r}")
            lines.append("")
        mitigations = ethics.get("mitigations", [])
        if mitigations:
            lines.append("### Mitigations")
            for m in mitigations:
                lines.append(f"- {m}")
            lines.append("")

        # Limitations
        if limitations:
            lines.append("## Limitations")
            for lim in limitations:
                lines.append(f"- {lim}")
            lines.append("")

        # Fairness Analysis
        if fairness.get("evaluated"):
            lines.append("## Fairness Analysis")
            lines.append(f"**Methodology:** {fairness.get('methodology', 'N/A')}")
            lines.append("")
            findings = fairness.get("findings", [])
            if findings:
                lines.append("**Findings:**")
                for f_item in findings:
                    lines.append(f"- {f_item}")
                lines.append("")
            fm = fairness.get("mitigations", [])
            if fm:
                lines.append("**Mitigations:**")
                for m in fm:
                    lines.append(f"- {m}")
                lines.append("")

        # Regulatory Status
        lines.append("## Regulatory Status")
        lines.append(f"- **Classification:** {regulatory.get('classification', 'N/A')}")
        lines.append(f"- **Conformity:** {regulatory.get('conformity_status', 'N/A')}")
        lines.append(f"- **Surveillance:** {regulatory.get('post_market_surveillance', 'N/A')}")
        lines.append("")
        regs = regulatory.get("applicable_regulations", [])
        if regs:
            lines.append("**Applicable Regulations:**")
            for r in regs:
                lines.append(f"- {r}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_generator: ModelCardAutoGenerator | None = None


def init_model_card_generator() -> None:
    """Create the module-level singleton and wire it to the event bus.

    Safe to call multiple times -- subsequent calls are no-ops.
    """
    global _generator
    if _generator is not None:
        logger.debug("Model card generator already initialised")
        return

    _generator = ModelCardAutoGenerator()

    # Subscribe to MODEL_PROMOTED if auto-generation is enabled
    if _generator._auto_generate:
        try:
            from src.agents.event_bus import EventType, event_bus

            event_bus.subscribe(
                EventType.MODEL_PROMOTED,
                _generator.on_model_promoted,
            )
            logger.info("Model card generator subscribed to MODEL_PROMOTED events")
        except Exception:
            logger.warning(
                "Could not subscribe to event bus -- "
                "model cards will not auto-generate on promotion",
                exc_info=True,
            )

    logger.info(
        "Model card generator singleton created (auto_generate=%s)",
        _generator._auto_generate,
    )


def get_model_card_generator() -> ModelCardAutoGenerator | None:
    """Return the module-level singleton, or ``None`` if not initialised."""
    return _generator
