"""Focused tests for the train.py model factory."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import torch

import train
import src.models.vignn as vignn_module


def test_build_model_vignn_passes_clinical_knowledge_graph(monkeypatch):
    captured = {}

    def fake_create_vignn_model(**kwargs):
        captured.update(kwargs)
        return torch.nn.Linear(4, 4)

    monkeypatch.setattr(vignn_module, "create_vignn_model", fake_create_vignn_model)

    cfg = {
        "model": {
            "name": "vignn",
            "num_classes": 2,
            "hidden_dim": 384,
            "num_graph_layers": 1,
            "num_heads": 2,
            "dropout": 0.1,
            "num_patches": 196,
            "patch_embed_dim": 384,
            "pretrained_backbone": False,
            "disease_names": ["DR", "ARMD"],
        }
    }

    train.build_model(cfg)

    assert "clinical_knowledge_graph" in captured
    assert "clinical_clinical_knowledge_graph" not in captured
