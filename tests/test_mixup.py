"""Tests for batch-level MixUp/CutMix config wiring."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

from src.data.mixup import MixUpCutMix, build_mixup


def test_build_mixup_reads_nested_train_config():
    cfg = {
        "augmentation": {
            "train": {
                "mixup_alpha": 0.2,
                "cutmix_alpha": 1.0,
                "mixup_cutmix_prob": 0.4,
            }
        }
    }

    mixup = build_mixup(cfg)

    assert isinstance(mixup, MixUpCutMix)
    assert mixup.mixup_alpha == 0.2
    assert mixup.cutmix_alpha == 1.0
    assert mixup.prob == 0.4
