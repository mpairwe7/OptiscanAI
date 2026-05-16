"""
RetinalDataModule - Handles data download, splitting, and DataLoader creation.
Extracted from notebook cells 0, 19-23.
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import RetinalDiseaseDataset
from src.data.validation import DataValidator

logger = logging.getLogger(__name__)

# Standard 48 disease columns from RFMiD
DISEASE_COLUMNS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL",
]


def _label_columns_from_df(df: pd.DataFrame) -> list[str]:
    exclude = {"ID", "Disease_Risk", "split", "original_split"}
    return [c for c in df.columns if c not in exclude]


def build_multilabel_stratify_labels(
    df: pd.DataFrame, disease_columns: list[str]
) -> np.ndarray | None:
    """Build stratification labels that reflect multilabel combinations."""
    if df.empty or not disease_columns:
        return None

    labels = (
        df[disease_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values.astype(np.float32)
    )
    label_counts = labels.sum(axis=1).astype(int)
    disease_risk = (
        df["Disease_Risk"].fillna("unknown").astype(str).values
        if "Disease_Risk" in df.columns
        else np.full(len(df), "unknown", dtype=object)
    )

    signatures = []
    for row, count, risk in zip(labels, label_counts, disease_risk):
        positive_indices = np.flatnonzero(row > 0.5)
        if len(positive_indices) == 0:
            signature = "__none__"
        else:
            codes = [disease_columns[i] for i in positive_indices[:4]]
            signature = "|".join(codes)
            if len(positive_indices) > 4:
                signature = f"{signature}|+{len(positive_indices) - 4}"
        signatures.append(f"{risk}::k={count}::{signature}")

    signature_counts = Counter(signatures)
    fallback_labels = [
        f"{risk}::k={count}" for risk, count in zip(disease_risk, label_counts)
    ]
    merged = [
        sig if signature_counts[sig] >= 2 else fallback
        for sig, fallback in zip(signatures, fallback_labels)
    ]
    merged_counts = Counter(merged)
    stratify_labels = [
        label if merged_counts[label] >= 2 else "__other__" for label in merged
    ]

    counts = Counter(stratify_labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    return np.asarray(stratify_labels, dtype=object)


class RetinalDataModule:
    """Manages RFMiD dataset download, preprocessing, and dataloaders."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self.img_size = self.data_cfg.get("img_size", 224)
        self.batch_size = self.data_cfg.get("batch_size", 32)
        self.num_workers = self.data_cfg.get("num_workers", 4)
        self.pin_memory = self.data_cfg.get("pin_memory", True)
        self.prefetch_factor = self.data_cfg.get("prefetch_factor", 2)

        # Resolve data directory
        data_dir = self.data_cfg.get("data_dir")
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None  # Resolved during prepare_data

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.disease_columns = None
        self.pos_weights = None

    def prepare_data(self):
        """Download dataset if not present (called on rank 0 only)."""
        if self.data_dir and self._data_ready():
            logger.info(f"Data already present at {self.data_dir}")
            return

        logger.info("Downloading RFMiD dataset via kagglehub...")
        import kagglehub

        dataset_name = self.data_cfg.get(
            "dataset_name",
            "mpairwelauben/multi-disease-retinal-eye-disease-dataset",
        )
        path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to: {path}")

        # Resolve the actual data root inside the download
        self.data_dir = self._resolve_data_root(Path(path))
        logger.info(f"Resolved data root: {self.data_dir}")

    def _resolve_data_root(self, path: Path) -> Path:
        """Find the actual RFMiD data root within the downloaded directory."""
        # Look for the RFMiD_All_Classes_Dataset pattern
        for candidate in path.rglob("*RFMiD*"):
            if candidate.is_dir() and "Groundtruths" in [
                d.name for d in candidate.iterdir() if d.is_dir()
            ]:
                return candidate
            # Also check one level deeper
            for sub in candidate.iterdir():
                if sub.is_dir() and "Groundtruths" in sub.name:
                    return candidate

        # Fallback: find directory containing CSV files
        for csv in path.rglob("*.csv"):
            if "RFMiD" in csv.name:
                return csv.parent.parent  # Go up from Groundtruths/
        return path

    def _data_ready(self) -> bool:
        """Check if data is already downloaded and organized."""
        if self.data_dir is None:
            return False
        csv_candidates = list(self.data_dir.rglob("*.csv"))
        img_candidates = list(self.data_dir.rglob("*.png")) + list(
            self.data_dir.rglob("*.jpg")
        )
        return len(csv_candidates) > 0 and len(img_candidates) > 0

    def setup(self, stage: str = "fit"):
        """Load data, create splits, and build datasets."""
        if self.data_dir is None:
            self.prepare_data()

        # Find label files
        label_files = sorted(self.data_dir.rglob("*.csv"))
        if not label_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        logger.info(f"Found label files: {[f.name for f in label_files]}")

        # Load labels per split
        train_labels, val_labels, test_labels = self._load_split_labels(label_files)
        logger.info(
            f"Loaded - Train: {len(train_labels)}, Val: {len(val_labels)}, "
            f"Test: {len(test_labels)}"
        )

        # Determine disease columns from the CSV
        exclude = {"ID", "Disease_Risk", "split", "original_split"}
        sample_df = train_labels if len(train_labels) > 0 else val_labels
        available_cols = [c for c in sample_df.columns if c not in exclude]
        self.disease_columns = [c for c in DISEASE_COLUMNS if c in available_cols]
        if not self.disease_columns:
            self.disease_columns = available_cols
        logger.info(f"Disease columns: {len(self.disease_columns)}")

        # Find image directories per split
        train_img_dir = self._find_image_dir("train", train_labels)
        val_img_dir = (
            self._find_image_dir("val", val_labels) if len(val_labels) > 0 else None
        )
        test_img_dir = (
            self._find_image_dir("test", test_labels) if len(test_labels) > 0 else None
        )
        logger.info(
            f"Image dirs - Train: {train_img_dir}, Val: {val_img_dir}, "
            f"Test: {test_img_dir}"
        )

        self._validate_splits(
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            train_img_dir=train_img_dir,
            val_img_dir=val_img_dir,
            test_img_dir=test_img_dir,
        )

        # Build transforms
        train_transform = get_train_transforms(self.cfg)
        val_transform = get_val_transforms(self.cfg)

        # Create datasets
        if stage in ("fit", None):
            self.train_dataset = RetinalDiseaseDataset(
                train_labels, train_img_dir, self.disease_columns, train_transform
            )
            self.val_dataset = RetinalDiseaseDataset(
                val_labels,
                val_img_dir if val_img_dir is not None else train_img_dir,
                self.disease_columns,
                val_transform,
            )
            self.pos_weights = self.train_dataset.get_pos_weights()

        if stage in ("test", None):
            if len(test_labels) == 0:
                self.test_dataset = None
                return
            self.test_dataset = RetinalDiseaseDataset(
                test_labels,
                test_img_dir if test_img_dir is not None else train_img_dir,
                self.disease_columns,
                val_transform,
            )

    def _load_split_labels(
        self, label_files: list[Path]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load CSV label files and assign to train/val/test splits."""
        split_map = {"train": None, "val": None, "test": None}

        for f in label_files:
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")  # Handle BOM
                if "ID" not in df.columns:
                    continue
                fname = f.stem.lower()
                if "train" in fname:
                    split_map["train"] = df
                elif "val" in fname or "evaluation" in fname:
                    split_map["val"] = df
                elif "test" in fname:
                    split_map["test"] = df
                logger.info(f"  Loaded {f.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Skipping {f.name}: {e}")

        # If any split is missing, merge and re-split
        if any(split_map[name] is None for name in ("train", "val", "test")):
            all_dfs = [v for v in split_map.values() if v is not None]
            if not all_dfs:
                raise ValueError("No valid label files found")
            merged = pd.concat(all_dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=["ID"], keep="first")

            label_columns = _label_columns_from_df(merged)
            train_ratio = float(self.data_cfg.get("train_split", 0.7))
            val_ratio = float(self.data_cfg.get("val_split", 0.2))
            test_ratio = float(self.data_cfg.get("test_split", 0.1))
            remainder = val_ratio + test_ratio
            if remainder <= 0:
                raise ValueError("val_split + test_split must be > 0")

            logger.warning(
                "One or more split files were missing. Rebuilding train/val/test "
                "using multilabel-aware stratification."
            )

            stratify = build_multilabel_stratify_labels(merged, label_columns)
            train_df, temp = train_test_split(
                merged,
                train_size=train_ratio,
                random_state=42,
                stratify=stratify,
            )

            temp_stratify = build_multilabel_stratify_labels(temp, label_columns)
            val_df, test_df = train_test_split(
                temp,
                train_size=val_ratio / remainder,
                random_state=42,
                stratify=temp_stratify,
            )
            return train_df, val_df, test_df

        train_df = split_map["train"]
        val_df = split_map["val"] if split_map["val"] is not None else pd.DataFrame()
        test_df = split_map["test"] if split_map["test"] is not None else pd.DataFrame()

        # Drop duplicates
        train_df = train_df.drop_duplicates(subset=["ID"], keep="first")
        if len(val_df) > 0:
            val_df = val_df.drop_duplicates(subset=["ID"], keep="first")
        if len(test_df) > 0:
            test_df = test_df.drop_duplicates(subset=["ID"], keep="first")

        return train_df, val_df, test_df

    def _find_image_dir(self, split: str = "train", labels_df: pd.DataFrame | None = None) -> Path:
        """Find image directory for a given split."""
        # Map split to directory name patterns
        patterns = {
            "train": ["*Training*", "*train*", "*Train*"],
            "val": ["*Validation*", "*val*", "*Val*", "*Evaluation*"],
            "test": ["*Testing*", "*test*", "*Test*"],
        }

        # Search for "1. Original Images" subdirectory first (RFMiD layout)
        img_root = None
        for candidate in self.data_dir.rglob("*Original Images*"):
            if candidate.is_dir():
                img_root = candidate
                break

        search_base = img_root if img_root else self.data_dir

        for pattern in patterns.get(split, patterns["train"]):
            for d in search_base.rglob(pattern):
                if d.is_dir():
                    img_count = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
                    if img_count > 0:
                        return d

        shared_dir = self._find_shared_image_dir(labels_df)
        if shared_dir is not None:
            logger.warning(f"Using shared image dir for {split}: {shared_dir}")
            return shared_dir

        raise FileNotFoundError(
            f"No image directory found for split '{split}' under {self.data_dir}"
        )

    def _find_shared_image_dir(self, labels_df: pd.DataFrame | None) -> Path | None:
        """Find a common image directory by checking whether sample IDs exist."""
        if labels_df is None or labels_df.empty:
            return None

        sample_ids = labels_df["ID"].astype(str).head(32).tolist()
        if not sample_ids:
            return None

        valid_exts = (".png", ".jpg", ".jpeg")
        best_dir = None
        best_hits = 0

        for candidate in self.data_dir.rglob("*"):
            if not candidate.is_dir():
                continue
            hits = 0
            for img_id in sample_ids:
                if any((candidate / f"{img_id}{ext}").exists() for ext in valid_exts):
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_dir = candidate
            if hits == len(sample_ids):
                return candidate

        return best_dir if best_hits == len(sample_ids) else None

    def _validate_splits(
        self,
        train_labels: pd.DataFrame,
        val_labels: pd.DataFrame,
        test_labels: pd.DataFrame,
        train_img_dir: Path,
        val_img_dir: Path | None,
        test_img_dir: Path | None,
    ):
        """Run configured validation checks and detect split leakage."""
        validation_cfg = self.cfg.get("data_validation", {})

        if validation_cfg.get("check_label_leakage", True):
            split_ids = {
                "train": set(train_labels["ID"].astype(str)),
                "val": set(val_labels["ID"].astype(str)) if len(val_labels) > 0 else set(),
                "test": set(test_labels["ID"].astype(str)) if len(test_labels) > 0 else set(),
            }
            overlaps = {
                "train_val": split_ids["train"] & split_ids["val"],
                "train_test": split_ids["train"] & split_ids["test"],
                "val_test": split_ids["val"] & split_ids["test"],
            }
            leaked = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
            if leaked:
                logger.warning(
                    f"Overlapping IDs across splits (may be expected if images are in "
                    f"separate directories with shared numeric IDs): {leaked}"
                )

        validator = DataValidator(disease_columns=self.disease_columns)
        split_entries = [
            ("train", train_labels, train_img_dir),
            ("val", val_labels, val_img_dir),
            ("test", test_labels, test_img_dir),
        ]
        validate_images = validation_cfg.get("validate_images", False)

        for split_name, labels_df, img_dir in split_entries:
            if len(labels_df) == 0:
                continue
            report = validator.validate_all(
                labels_df,
                img_dir=img_dir if validate_images else None,
            )
            for result in report.results:
                message = (
                    f"[{split_name}] {result.check_name}: {result.details}"
                )
                if result.passed:
                    logger.info(message)
                elif result.severity == "warning":
                    logger.warning(message)
                else:
                    raise ValueError(message)

    def _loader_kwargs(self) -> dict:
        """Common DataLoader kwargs, handling num_workers=0 case."""
        kwargs = {
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory and self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["num_workers"] = self.num_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = True
        else:
            kwargs["num_workers"] = 0
        return kwargs

    def train_dataloader(self, distributed: bool = False) -> DataLoader:
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if distributed else None
        return DataLoader(
            self.train_dataset,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self, distributed: bool = False) -> DataLoader:
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if distributed else None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )

    def test_dataloader(self, distributed: bool = False) -> DataLoader:
        if self.test_dataset is None:
            return None
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if distributed else None
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )
