#!/usr/bin/env python3
"""
Main training entry point for retinal disease classification.

Launch with torchrun for multi-GPU DDP training:
    torchrun --nproc_per_node=8 train.py --config configs/train.yaml

Or use the convenience script:
    bash scripts/train_multigpu.sh
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml

from src.data.datamodule import RetinalDataModule
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
    """Initialize DDP process group."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def set_seed(seed: int):
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg: dict) -> torch.nn.Module:
    """Build model from config."""
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    num_classes = model_cfg["num_classes"]

    # Build knowledge graph (shared by all 4 custom models)
    def _build_kg():
        from src.data.datamodule import DISEASE_COLUMNS
        from src.models.vignn import ClinicalKnowledgeGraph

        names = model_cfg.get("disease_names")
        if not names:
            names = (
                DISEASE_COLUMNS[:num_classes]
                if num_classes <= len(DISEASE_COLUMNS)
                else DISEASE_COLUMNS
            )
        return ClinicalKnowledgeGraph(disease_names=names)

    hidden = model_cfg.get("hidden_dim", 384)
    heads = model_cfg.get("num_heads", 4)
    layers = model_cfg.get("num_graph_layers", 3)
    drop = model_cfg.get("dropout", 0.1)

    if model_name == "vignn":
        from src.models.vignn import create_vignn_model

        kg = _build_kg()
        model = create_vignn_model(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_graph_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
            num_patches=model_cfg.get("num_patches", 196),
            patch_embed_dim=model_cfg.get("patch_embed_dim", 384),
        )

    elif model_name == "graphclip":
        from src.models.graphclip import GraphCLIP

        kg = _build_kg()
        model = GraphCLIP(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_graph_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
        )

    elif model_name == "visual_language_gnn":
        from src.models.visual_language_gnn import VisualLanguageGNN

        kg = _build_kg()
        model = VisualLanguageGNN(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
        )

    elif model_name == "scene_graph_transformer":
        from src.models.scene_graph_transformer import SceneGraphTransformer

        kg = _build_kg()
        model = SceneGraphTransformer(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
        )

    elif model_name == "hybrid":
        from src.models.retinal_foundation_hybrid import create_hybrid_model

        kg = _build_kg()
        model = create_hybrid_model(
            num_classes=num_classes,
            hidden_dim=hidden,
            num_graph_layers=layers,
            num_heads=heads,
            dropout=drop,
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

    elif model_name == "vit":
        import timm

        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=model_cfg.get("pretrained_backbone", True),
            num_classes=num_classes,
        )

    elif model_name == "efficientnet":
        import timm

        model = timm.create_model(
            "efficientnet_b4",
            pretrained=model_cfg.get("pretrained_backbone", True),
            num_classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: hybrid, vignn, graphclip, visual_language_gnn, scene_graph_transformer, vit, efficientnet"
        )

    # Log param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Model: {model_name} | Total: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M"
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train retinal disease model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    setup_logging(rank)
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("training", {}).get("seed", 42))

    if rank == 0:
        logger.info(f"Config loaded from {args.config}")
        logger.info(f"World size: {world_size} GPUs")
        if torch.cuda.is_available():
            for i in range(min(world_size, torch.cuda.device_count())):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / 1e9
                logger.info(f"  GPU {i}: {name} ({mem:.0f} GB)")

    # Data
    datamodule = RetinalDataModule(cfg)
    if rank == 0:
        datamodule.prepare_data()

    # Barrier: wait for rank 0 to finish downloading
    if world_size > 1:
        dist.barrier()

    datamodule.setup(stage="fit")

    # Update num_classes from actual data
    actual_classes = len(datamodule.disease_columns)
    if cfg["model"]["num_classes"] != actual_classes:
        logger.info(
            f"Adjusting num_classes from {cfg['model']['num_classes']} to {actual_classes} "
            f"(from data)"
        )
        cfg["model"]["num_classes"] = actual_classes
    cfg["model"]["disease_names"] = datamodule.disease_columns

    # Model
    model = build_model(cfg)

    # Loss with class weights
    pos_weight = datamodule.pos_weights.to(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    criterion = build_loss(cfg, pos_weight=pos_weight)

    # Trainer
    trainer = DDPTrainer(
        model=model,
        criterion=criterion,
        cfg=cfg,
        datamodule=datamodule,
    )

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}", weights_only=False)
        raw_model = trainer.model.module if trainer.distributed else trainer.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Train
    trainer.train()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        logger.info("Done.")


if __name__ == "__main__":
    main()
