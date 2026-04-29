# Data Augmentation

## Strategy

Medical imaging augmentations optimized for retinal fundus photographs. Configured entirely via `configs/train.yaml` (legacy) or `configs/hybrid_precision_2026.yaml` (v2).

## V2 Test-Time Augmentation (TTA)

At inference, the v2 model averages predictions over 6 augmented views:
1. Original image
2. Horizontal flip
3. Vertical flip
4. 90-degree rotation
5. 180-degree rotation
6. 270-degree rotation

This provides a 2-4% F1 boost with no training cost. Configured via `tta.n_augments: 6` in the config. See `RetinalFoundationHybridV2.predict_with_tta()` in `src/models/retinal_foundation_hybrid_v2.py`.

## Training Augmentations

```yaml
augmentation:
  train:
    random_horizontal_flip: 0.5
    random_vertical_flip: 0.5
    random_rotation: 15          # degrees
    color_jitter:
      brightness: 0.25
      contrast: 0.25
      saturation: 0.15
      hue: 0.05
    random_resized_crop:
      scale: [0.75, 1.0]
      ratio: [0.9, 1.1]
    random_erasing: 0.15
    normalize:
      mean: [0.485, 0.456, 0.406]   # ImageNet stats
      std: [0.229, 0.224, 0.225]
```

### Pipeline Order

1. `RandomResizedCrop(224)` - Scale/crop variation
2. `RandomHorizontalFlip(0.5)` - Fundus orientation invariance
3. `RandomVerticalFlip(0.5)` - Rotational invariance
4. `RandomRotation(15)` - Slight tilt compensation
5. `ColorJitter(...)` - Lighting/camera variation
6. `ToTensor()` - Convert to [0,1] float tensor
7. `Normalize(ImageNet)` - Match pretrained backbone stats
8. `RandomErasing(0.15)` - Occlusion robustness

### Validation/Test Pipeline

Only resize + normalize (no augmentation):

```yaml
augmentation:
  val:
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

## Medical Imaging Considerations

- **No extreme color shifts**: Retinal pathology colors (hemorrhages, exudates) are diagnostically important
- **Conservative rotation**: Fundus images have consistent orientation
- **Random erasing**: Simulates optic artifacts/occlusions common in clinical settings
- **Scale variation**: Accounts for different fundus camera magnifications

## Implementation

`src/data/augmentation.py` provides:
- `get_train_transforms(cfg)` - Builds training pipeline from YAML config
- `get_val_transforms(cfg)` - Builds validation pipeline

All transforms use `torchvision.transforms` for GPU compatibility during training.
