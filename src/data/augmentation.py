"""
Data augmentation pipelines for retinal fundus images.
Extracted and enhanced from notebook cell 22.
"""

import torchvision.transforms as T


def get_train_transforms(cfg: dict) -> T.Compose:
    """Build training augmentation pipeline from config."""
    aug = cfg.get("augmentation", {}).get("train", {})
    norm = aug.get("normalize", {})
    img_size = cfg.get("data", {}).get("img_size", 224)

    transforms_list = []

    # Random resized crop
    rrc = aug.get("random_resized_crop", {})
    if rrc:
        transforms_list.append(
            T.RandomResizedCrop(
                img_size,
                scale=tuple(rrc.get("scale", [0.8, 1.0])),
                ratio=tuple(rrc.get("ratio", [0.9, 1.1])),
                interpolation=T.InterpolationMode.BICUBIC,
            )
        )
    else:
        transforms_list.append(T.Resize((img_size, img_size)))

    # Flips
    if aug.get("random_horizontal_flip", 0) > 0:
        transforms_list.append(T.RandomHorizontalFlip(p=aug["random_horizontal_flip"]))
    if aug.get("random_vertical_flip", 0) > 0:
        transforms_list.append(T.RandomVerticalFlip(p=aug["random_vertical_flip"]))

    # Rotation
    if aug.get("random_rotation", 0) > 0:
        transforms_list.append(T.RandomRotation(degrees=aug["random_rotation"], fill=0))

    # Color jitter
    cj = aug.get("color_jitter", {})
    if cj:
        transforms_list.append(
            T.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            )
        )

    # To tensor + normalize
    transforms_list.append(T.ToTensor())

    if norm:
        transforms_list.append(T.Normalize(mean=norm["mean"], std=norm["std"]))

    # Random erasing (after ToTensor)
    if aug.get("random_erasing", 0) > 0:
        transforms_list.append(T.RandomErasing(p=aug["random_erasing"]))

    return T.Compose(transforms_list)


def get_val_transforms(cfg: dict) -> T.Compose:
    """Build validation/test transform pipeline from config."""
    aug = cfg.get("augmentation", {}).get("val", {})
    norm = aug.get("normalize", {})
    img_size = cfg.get("data", {}).get("img_size", 224)

    transforms_list = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ]

    if norm:
        transforms_list.append(T.Normalize(mean=norm["mean"], std=norm["std"]))

    return T.Compose(transforms_list)


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------
import torch


def get_tta_transforms(cfg: dict) -> list[T.Compose]:
    """Build TTA transform pipelines (original + geometric variants)."""
    base = get_val_transforms(cfg)
    aug = cfg.get("augmentation", {}).get("val", {})
    norm = aug.get("normalize", {})
    img_size = cfg.get("data", {}).get("img_size", 224)

    tta_names = cfg.get("augmentation", {}).get("tta", {}).get("transforms", ["hflip", "vflip"])

    def _build(extra):
        parts = [T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC)]
        parts.extend(extra)
        parts.append(T.ToTensor())
        if norm:
            parts.append(T.Normalize(mean=norm["mean"], std=norm["std"]))
        return T.Compose(parts)

    tta_map = {
        "hflip": [T.RandomHorizontalFlip(p=1.0)],
        "vflip": [T.RandomVerticalFlip(p=1.0)],
        "rotate90": [T.Lambda(lambda img: img.rotate(90))],
        "rotate180": [T.Lambda(lambda img: img.rotate(180))],
        "rotate270": [T.Lambda(lambda img: img.rotate(270))],
    }

    pipelines = [base]
    for name in tta_names:
        if name in tta_map:
            pipelines.append(_build(tta_map[name]))
    return pipelines


class TTAPredictor:
    """Averages logits across TTA variants for better calibrated predictions."""

    def __init__(self, model, transforms: list[T.Compose], device):
        self.model = model
        self.transforms = transforms
        self.device = device

    @torch.no_grad()
    def predict(self, pil_image) -> torch.Tensor:
        self.model.eval()
        all_logits = []
        for tfm in self.transforms:
            tensor = tfm(pil_image).unsqueeze(0).to(self.device)
            all_logits.append(self.model(tensor))
        return torch.sigmoid(torch.stack(all_logits).mean(dim=0)).cpu()
