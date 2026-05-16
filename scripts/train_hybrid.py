#!/usr/bin/env python3
"""
Training entry point for RetinalFoundationHybrid.

Usage:
    # Single GPU
    python scripts/train_hybrid.py --config configs/hybrid_2026.yaml

    # Multi-GPU DDP
    torchrun --nproc_per_node=8 scripts/train_hybrid.py --config configs/hybrid_2026.yaml

    # With LoRA rank override
    torchrun --nproc_per_node=8 scripts/train_hybrid.py --config configs/hybrid_2026.yaml --lora-rank 32
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import yaml

from src.data.datamodule import RetinalDataModule
from src.models.retinal_foundation_hybrid import create_hybrid_model
from src.models.vignn import create_knowledge_graph
from src.training.losses import build_loss
from src.training.trainer import DDPTrainer


def setup_logging(rank: int = 0):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[Rank {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_hybrid_model(cfg: dict) -> torch.nn.Module:
    """Build RetinalFoundationHybrid from config."""
    model_cfg = cfg["model"]
    num_classes = model_cfg["num_classes"]

    # Build knowledge graph
    from src.data.datamodule import DISEASE_COLUMNS
    disease_names = model_cfg.get("disease_names")
    if not disease_names:
        disease_names = (
            DISEASE_COLUMNS[:num_classes]
            if num_classes <= len(DISEASE_COLUMNS)
            else DISEASE_COLUMNS
        )
    kg = create_knowledge_graph(disease_names=disease_names)

    model = create_hybrid_model(
        num_classes=num_classes,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_graph_layers=model_cfg.get("num_graph_layers", 2),
        num_heads=model_cfg.get("num_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
        clinical_knowledge_graph=kg,
        backbone=model_cfg.get("backbone", "vit_large_patch16_224"),
        img_size=model_cfg.get("img_size", 224),
        use_lora=model_cfg.get("use_lora", True),
        lora_rank=model_cfg.get("lora_rank", 16),
        lora_alpha=model_cfg.get("lora_alpha", 16.0),
        num_ensemble_heads=model_cfg.get("num_ensemble_heads", 3),
        mc_dropout=model_cfg.get("mc_dropout", 0.15),
        enable_moe=model_cfg.get("enable_moe", True),
        moe_top_k=model_cfg.get("moe_top_k", 2),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
    )

    params = model.get_param_summary()
    logger = logging.getLogger(__name__)
    logger.info(
        f"Hybrid model | Total: {params['total']/1e6:.1f}M | "
        f"Trainable: {params['trainable']/1e6:.1f}M"
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train RetinalFoundationHybrid")
    parser.add_argument("--config", type=str, default="configs/hybrid_2026.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone")
    parser.add_argument("--no-moe", action="store_true", help="Disable MoE")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    setup_logging(rank)
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.lora_rank is not None:
        cfg["model"]["lora_rank"] = args.lora_rank
    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone
    if args.no_moe:
        cfg["model"]["enable_moe"] = False

    set_seed(cfg.get("training", {}).get("seed", 42))

    if rank == 0:
        logger.info(f"Config: {args.config}")
        logger.info(f"World size: {world_size} GPUs")
        logger.info(f"Backbone: {cfg['model'].get('backbone', 'vit_large_patch16_224')}")
        logger.info(f"LoRA rank: {cfg['model'].get('lora_rank', 16)}")
        logger.info(f"MoE: {cfg['model'].get('enable_moe', True)}")

    # Data
    datamodule = RetinalDataModule(cfg)
    if rank == 0:
        datamodule.prepare_data()
    if world_size > 1:
        dist.barrier()
    datamodule.setup(stage="fit")

    # Adjust num_classes
    actual_classes = len(datamodule.disease_columns)
    if cfg["model"]["num_classes"] != actual_classes:
        logger.info(f"Adjusting num_classes: {cfg['model']['num_classes']} -> {actual_classes}")
        cfg["model"]["num_classes"] = actual_classes
    cfg["model"]["disease_names"] = datamodule.disease_columns

    # Model
    model = build_hybrid_model(cfg)

    # Loss
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    pos_weight = datamodule.pos_weights.to(device)
    criterion = build_loss(cfg, pos_weight=pos_weight)

    # Trainer
    trainer = DDPTrainer(
        model=model,
        criterion=criterion,
        cfg=cfg,
        datamodule=datamodule,
    )

    # Resume
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = trainer.model.module if trainer.distributed else trainer.model
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Train
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
