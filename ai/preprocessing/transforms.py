"""Image transforms for training and for evaluation.

Two builders, deliberately kept separate: augmentation exists only in the
training pipeline. Validation and test images go through a fixed, deterministic
resize/crop so that a metric change reflects the model, never the sampling of a
random crop.
"""

from torchvision import transforms

from ai.preprocessing.config import DataConfig


def _normalize(config: DataConfig) -> transforms.Normalize:
    return transforms.Normalize(
        mean=list(config.normalization.mean),
        std=list(config.normalization.std),
    )


def build_train_transforms(config: DataConfig) -> transforms.Compose:
    """Augmented pipeline for the training split.

    Dermatoscopic images have no canonical orientation - the dermatoscope is
    placed on the skin at an arbitrary angle - so flips and rotation are label
    preserving. Colour jitter is kept mild because pigmentation and erythema
    are diagnostic signals, not nuisance variation.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            _normalize(config),
        ]
    )


def build_eval_transforms(config: DataConfig) -> transforms.Compose:
    """Deterministic pipeline for the validation and test splits.

    Resize the shorter edge then centre-crop, matching the standard ImageNet
    evaluation protocol the pretrained weights were validated under.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.resize_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            _normalize(config),
        ]
    )
