"""End-to-end: manifests on disk -> model-ready batches."""

import pandas as pd
import pytest
import torch

from ai.preprocessing.datamodule import build_dataloaders
from ai.preprocessing.dataset import HAM10000Dataset, load_manifest
from ai.preprocessing.labels import NUM_CLASSES
from ai.preprocessing.sampling import (
    build_weighted_sampler,
    class_counts,
    compute_class_weights,
)
from ai.preprocessing.splits import SPLIT_NAMES, lesion_overlap
from ai.preprocessing.transforms import build_eval_transforms


def test_prepare_writes_one_manifest_per_split(prepared_config):
    for name in SPLIT_NAMES:
        assert prepared_config.manifest_path(name).is_file()


def test_written_manifests_have_zero_lesion_overlap(prepared_config):
    frames = [
        load_manifest(prepared_config.manifest_path(name)) for name in SPLIT_NAMES
    ]
    assert lesion_overlap(frames) == set()


def test_batches_are_model_ready(prepared_config):
    loaders = build_dataloaders(prepared_config)
    size = prepared_config.image_size

    for loader in (loaders.train, loaders.val, loaders.test):
        images, targets = next(iter(loader))
        assert images.shape[1:] == (3, size, size)
        assert images.dtype == torch.float32
        assert targets.dtype == torch.int64
        assert images.shape[0] == targets.shape[0]
        assert int(targets.min()) >= 0
        assert int(targets.max()) < NUM_CLASSES


def test_loaders_cover_their_split_exactly(prepared_config):
    loaders = build_dataloaders(prepared_config)
    for name, loader in (
        ("train", loaders.train),
        ("val", loaders.val),
        ("test", loaders.test),
    ):
        expected = len(load_manifest(prepared_config.manifest_path(name)))
        assert len(loader.dataset) == expected


def test_eval_loaders_are_not_shuffled_or_resampled(prepared_config):
    loaders = build_dataloaders(prepared_config, use_weighted_sampler=True)
    for loader in (loaders.val, loaders.test):
        assert loader.sampler is not None
        assert not isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
        first = [batch[1] for batch in loader]
        second = [batch[1] for batch in loader]
        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_weighted_sampler_is_opt_in(prepared_config):
    default = build_dataloaders(prepared_config)
    assert not isinstance(
        default.train.sampler, torch.utils.data.WeightedRandomSampler
    )

    opted_in = build_dataloaders(prepared_config, use_weighted_sampler=True)
    assert isinstance(
        opted_in.train.sampler, torch.utils.data.WeightedRandomSampler
    )


def test_train_uses_augmentation_and_eval_does_not(prepared_config):
    loaders = build_dataloaders(prepared_config)
    train_dataset = loaders.train.dataset

    torch.manual_seed(0)
    assert not torch.equal(train_dataset[0][0], train_dataset[0][0])

    val_dataset = loaders.val.dataset
    assert torch.equal(val_dataset[0][0], val_dataset[0][0])


def test_class_weights_favour_rare_classes(prepared_config):
    loaders = build_dataloaders(prepared_config)
    weights = compute_class_weights(loaders.train_labels)
    counts = class_counts(loaders.train_labels)

    assert weights.shape == (NUM_CLASSES,)
    assert pytest.approx(1.0, abs=1e-5) == float(weights.mean())

    rarest = int(counts.argmin())
    most_common = int(counts.argmax())
    assert weights[rarest] > weights[most_common]


def test_weighted_sampler_balances_draws(prepared_config):
    loaders = build_dataloaders(prepared_config)
    labels = loaders.train_labels
    counts = class_counts(labels)

    generator = torch.Generator().manual_seed(0)
    sampler = build_weighted_sampler(labels, generator=generator)
    assert len(sampler) == len(labels)

    drawn = class_counts([labels[i] for i in list(sampler)])
    rarest = int(counts.argmin())
    most_common = int(counts.argmax())
    # Rare classes are revisited: the sampled share of the rarest class rises
    # well above its share of the underlying split.
    assert drawn[rarest] / len(labels) > counts[rarest] / len(labels)
    assert drawn[most_common] / len(labels) < counts[most_common] / len(labels)


def test_missing_manifest_reports_how_to_generate_it(config):
    absent = config.model_copy(
        update={"manifest_dir": config.metadata_csv.parent / "absent"}
    )
    with pytest.raises(FileNotFoundError, match="ai.preprocessing.prepare"):
        build_dataloaders(absent)


def test_manifest_referencing_an_absent_image_is_rejected(prepared_config):
    manifest = load_manifest(prepared_config.manifest_path("val"))
    manifest.loc[0, "image_id"] = "ISIC_does_not_exist"

    with pytest.raises(FileNotFoundError, match="manifest images not found"):
        HAM10000Dataset(
            manifest,
            prepared_config.raw_image_dirs,
            build_eval_transforms(prepared_config),
        )


def test_manifest_columns_are_stable(prepared_config):
    manifest = pd.read_csv(prepared_config.manifest_path("train"))
    assert list(manifest.columns) == [
        "image_id",
        "lesion_id",
        "dx",
        "label_idx",
        "split",
    ]
