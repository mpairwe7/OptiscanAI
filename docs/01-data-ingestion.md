# Data Ingestion

## Dataset: RFMiD (Retinal Fundus Multi-disease Image Dataset)

| Property | Value |
|---|---|
| Source | KaggleHub: `mpairwelauben/multi-disease-retinal-eye-disease-dataset` |
| Train | 1,920 images |
| Validation | 640 images |
| Test | 640 images |
| Classes | 45 retinal diseases (multi-label) |
| **V2 filtered classes** | **25-28 classes** (ultra-rare <10 samples dropped) |
| Image format | PNG, variable resolution |
| Labels | CSV with binary columns per disease |

> **V2 Class Filtering**: Many classes have <5 training samples. The model cannot learn
> a decision boundary from 2-3 examples. `filter_rare_classes()` in
> `src/models/retinal_foundation_hybrid_v2.py` drops classes with <10 positive
> training samples, reducing from 45 to ~25-28 learnable classes.
> This is configured via `class_filtering.min_samples` in `configs/hybrid_precision_2026.yaml`.

## How It Works

Data ingestion is handled by `src/data/datamodule.py`:

```
RetinalDataModule
  .prepare_data()     -> Downloads from KaggleHub (rank 0 only in DDP)
  .setup("fit")       -> Loads CSVs, discovers image dirs, creates datasets
  .train_dataloader() -> Returns DataLoader with DistributedSampler
```

### Download

```bash
# Automatic (via pipeline)
PYTHONPATH=. python3 -c "
from src.data.datamodule import RetinalDataModule
import yaml
cfg = yaml.safe_load(open('configs/train.yaml'))
dm = RetinalDataModule(cfg)
dm.prepare_data()
"

# Manual
pip install kagglehub
python3 -c "import kagglehub; kagglehub.dataset_download('mpairwelauben/multi-disease-retinal-eye-disease-dataset')"
```

Data is cached at `~/.cache/kagglehub/datasets/mpairwelauben/...` and auto-discovered by the datamodule. No manual copying needed.

### Directory Layout (KaggleHub cache)

```
A. RFMiD_All_Classes_Dataset/
  1. Original Images/
    a. Training Set/       (1920 PNGs)
    b. Validation Set/     (640 PNGs)
    c. Testing Set/        (640 PNGs)
  2. Groundtruths/
    a. RFMiD_Training_Labels.csv
    b. RFMiD_Validation_Labels.csv
    c. RFMiD_Testing_Labels.csv
```

### Label Format

Each CSV has columns: `ID, Disease_Risk, DR, ARMD, MH, DN, MYA, BRVO, ...` (45 binary disease columns).

### Class Imbalance

The dataset has severe class imbalance (DR at 40% vs rare diseases at <0.5%). This is handled by:
- **Focal Loss / Asymmetric Loss** in `src/training/losses.py`
- **Pos-weight computation** in `src/data/dataset.py` (`get_pos_weights()`)
- **Data augmentation** (see `docs/02-data-augmentation.md`)

## Configuration

All data settings are in `configs/train.yaml` under the `data:` section:

```yaml
data:
  dataset_name: "mpairwelauben/multi-disease-retinal-eye-disease-dataset"
  data_dir: null          # Auto-resolved from kagglehub cache
  img_size: 224
  batch_size: 32          # Per-GPU (effective = 32 * 8 GPUs = 256)
  num_workers: 0          # Set 4+ when multiprocessing is available
```

## Key Files

| File | Purpose |
|---|---|
| `src/data/datamodule.py` | Download, split, DataLoader creation |
| `src/data/dataset.py` | `RetinalDiseaseDataset` PyTorch Dataset |
| `src/data/augmentation.py` | Transform pipelines |
| `configs/train.yaml` | Data configuration |
