"""Single entry point producing the three model-ready DataLoaders."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ai.preprocessing.config import DataConfig
from ai.preprocessing.dataset import HAM10000Dataset, load_manifest
from ai.preprocessing.sampling import build_weighted_sampler
from ai.preprocessing.transforms import build_eval_transforms, build_train_transforms


@dataclass(frozen=True)
class DataLoaders:
    """The three loaders, plus the training labels.

    `train_labels` is exposed so Phase 2.2 can derive loss weights via
    `sampling.compute_class_weights` without re-reading the manifest.
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_labels: list[int]


def build_dataloaders(
    config: DataConfig,
    use_weighted_sampler: bool = False,
) -> DataLoaders:
    """Build train/val/test loaders from the committed split manifests.

    Augmentation is applied to the training loader only. The evaluation loaders
    are never shuffled and never resampled, so val and test always report over
    the true class distribution in a stable order.

    `use_weighted_sampler` is off by default: see `sampling` for why the two
    imbalance strategies are not combined.
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_transform = build_train_transforms(config)
    eval_transform = build_eval_transforms(config)

    train_manifest = load_manifest(config.manifest_path("train"))
    train_dataset = HAM10000Dataset(
        train_manifest, config.raw_image_dirs, train_transform
    )
    train_labels = list(train_dataset.labels)

    sampler = (
        build_weighted_sampler(train_labels, generator=generator)
        if use_weighted_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        # A sampler and shuffle=True are mutually exclusive in torch.
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        generator=generator,
        drop_last=False,
    )

    eval_loaders = {
        split: DataLoader(
            HAM10000Dataset(
                load_manifest(config.manifest_path(split)),
                config.raw_image_dirs,
                eval_transform,
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        for split in ("val", "test")
    }

    return DataLoaders(
        train=train_loader,
        val=eval_loaders["val"],
        test=eval_loaders["test"],
        train_labels=train_labels,
    )
