"""Dataset Card / Datasheet generator following Gebru et al. (2021) framework."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetCard:
    # Motivation
    name: str = "RFMiD - Retinal Fundus Multi-Disease Image Dataset"
    purpose: str = "Multi-label retinal disease classification for automated screening"
    creators: str = (
        "Originally collected by RFMiD project team; curated for MLOps by project maintainers"
    )
    funding: str = ""

    # Composition
    num_samples: int = 0
    num_classes: int = 45
    data_format: str = "Fundus images (JPEG/PNG) + CSV labels"
    class_distribution: dict = field(default_factory=dict)
    multi_label_stats: dict = field(default_factory=dict)

    # Collection
    collection_process: str = "Retrospective collection from clinical fundus photography"
    time_period: str = "Not specified in source dataset"
    geography: str = "Multi-center, specific demographics not documented"
    consent: str = "De-identified clinical images, IRB/ethics details per source dataset"

    # Preprocessing
    preprocessing: str = "Resized to 224x224, normalized with ImageNet statistics"
    splitting: str = "Stratified random split: 70% train, 20% validation, 10% test"

    # Uses
    intended_use: str = "Training and evaluating retinal disease classification models for research"
    not_suitable_for: list[str] = field(
        default_factory=lambda: [
            "Clinical diagnosis without clinician oversight",
            "Populations not represented in the dataset",
            "Non-fundus imaging modalities",
        ]
    )

    # Distribution
    license: str = "CC-BY-4.0"
    access: str = "Kaggle: mpairwelauben/multi-disease-retinal-eye-disease-dataset"

    # Maintenance
    maintainer: str = ""
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    known_issues: list[str] = field(
        default_factory=lambda: [
            "Severe class imbalance - some diseases have <5 samples",
            "Demographic information (age, sex, ethnicity) not available per image",
            "Image quality varies across collection sites",
            "Some images may have been captured with different camera systems",
        ]
    )

    def populate_from_dataframe(self, df: pd.DataFrame, disease_columns: list[str]):
        """Auto-populate statistics from the dataset."""
        self.num_samples = len(df)
        self.num_classes = len(disease_columns)

        # Class distribution
        class_counts = df[disease_columns].sum().to_dict()
        self.class_distribution = {k: int(v) for k, v in class_counts.items()}

        # Multi-label stats
        labels_per_sample = df[disease_columns].sum(axis=1)
        self.multi_label_stats = {
            "mean_labels_per_sample": float(labels_per_sample.mean()),
            "max_labels_per_sample": int(labels_per_sample.max()),
            "min_labels_per_sample": int(labels_per_sample.min()),
            "samples_with_no_disease": int((labels_per_sample == 0).sum()),
            "samples_with_multiple_diseases": int((labels_per_sample > 1).sum()),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def to_markdown(self, path: Path):
        md = self._render_markdown()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md)

    def _render_markdown(self) -> str:
        lines = [
            f"# Dataset Card: {self.name}",
            f"*Last updated: {self.last_updated}*\n",
            "## Motivation",
            f"**Purpose:** {self.purpose}",
            f"**Creators:** {self.creators}\n",
            "## Composition",
            f"- **Samples:** {self.num_samples:,}",
            f"- **Classes:** {self.num_classes}",
            f"- **Format:** {self.data_format}\n",
        ]

        if self.multi_label_stats:
            lines += [
                "### Multi-label Statistics",
                f"- Mean labels per sample: {self.multi_label_stats.get('mean_labels_per_sample', 0):.2f}",
                f"- Max labels per sample: {self.multi_label_stats.get('max_labels_per_sample', 0)}",
                f"- Samples with no disease: {self.multi_label_stats.get('samples_with_no_disease', 0)}",
                f"- Samples with multiple diseases: {self.multi_label_stats.get('samples_with_multiple_diseases', 0)}\n",
            ]

        lines += [
            "## Collection",
            f"- **Process:** {self.collection_process}",
            f"- **Geography:** {self.geography}",
            f"- **Consent:** {self.consent}\n",
            "## Preprocessing",
            f"- {self.preprocessing}",
            f"- Split: {self.splitting}\n",
            "## Intended Use",
            f"{self.intended_use}\n",
            "### Not Suitable For",
        ]
        for ns in self.not_suitable_for:
            lines.append(f"- {ns}")

        lines += ["\n## Known Issues"]
        for issue in self.known_issues:
            lines.append(f"- {issue}")

        lines += [f"\n## License\n{self.license}", f"\n## Access\n{self.access}"]

        if self.class_distribution:
            lines += ["\n## Class Distribution (Top 20)"]
            sorted_classes = sorted(
                self.class_distribution.items(), key=lambda x: x[1], reverse=True
            )[:20]
            lines += ["| Disease | Count |", "|---------|-------|"]
            for disease, count in sorted_classes:
                lines.append(f"| {disease} | {count} |")

        return "\n".join(lines)
