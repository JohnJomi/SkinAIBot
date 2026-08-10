"""The split is lesion-grouped, class-stratified, and reproducible."""

import pandas as pd
import pytest

from ai.preprocessing.config import SplitRatios
from ai.preprocessing.labels import CLASS_CODES
from ai.preprocessing.splits import (
    SPLIT_NAMES,
    assign_splits,
    build_lesion_table,
    lesion_overlap,
    split_frames,
)

RATIOS = SplitRatios(train=0.70, val=0.15, test=0.15)


@pytest.fixture(scope="module")
def manifest(metadata: pd.DataFrame) -> pd.DataFrame:
    return assign_splits(metadata, RATIOS, seed=42)


@pytest.fixture(scope="module")
def frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return split_frames(manifest)


def test_zero_lesion_overlap_across_all_three_splits(frames):
    train, val, test = (set(frames[name]["lesion_id"]) for name in SPLIT_NAMES)

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert lesion_overlap([frames[name] for name in SPLIT_NAMES]) == set()


def test_every_image_of_a_lesion_lands_in_one_split(manifest):
    splits_per_lesion = manifest.groupby("lesion_id")["split"].nunique()
    assert (splits_per_lesion == 1).all()


def test_all_images_assigned_exactly_once(metadata, manifest, frames):
    assert len(manifest) == len(metadata)
    assert set(manifest["image_id"]) == set(metadata["image_id"])
    assert sum(len(frames[name]) for name in SPLIT_NAMES) == len(metadata)


def test_every_class_present_in_every_split(frames):
    for name in SPLIT_NAMES:
        assert set(frames[name]["dx"]) == set(CLASS_CODES), name


def test_class_distribution_approximately_preserved(metadata, frames):
    overall = metadata["dx"].value_counts(normalize=True)
    for name in SPLIT_NAMES:
        realised = frames[name]["dx"].value_counts(normalize=True)
        for code in CLASS_CODES:
            # Loose on a ~90-image fixture: one indivisible lesion is a large
            # fraction of a rare class here. The assertion that matters is that
            # no class collapses or explodes.
            assert abs(realised[code] - overall[code]) < 0.15, (name, code)


def test_split_sizes_near_target_ratios(manifest, frames):
    total = len(manifest)
    for name, expected in RATIOS.as_dict().items():
        assert abs(len(frames[name]) / total - expected) < 0.10, name


def test_same_seed_reproduces_the_split(metadata, manifest):
    repeated = assign_splits(metadata, RATIOS, seed=42)
    pd.testing.assert_frame_equal(manifest, repeated)


def test_split_is_independent_of_metadata_row_order(metadata, manifest):
    reordered = metadata.sort_values("image_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(assign_splits(reordered, RATIOS, seed=42), manifest)


def test_different_seed_changes_the_split(metadata, manifest):
    other = assign_splits(metadata, RATIOS, seed=1234)
    assert not other["split"].equals(manifest["split"])


def test_label_indices_match_the_canonical_mapping(manifest):
    for code, group in manifest.groupby("dx"):
        assert group["label_idx"].nunique() == 1
        assert group["label_idx"].iloc[0] == CLASS_CODES.index(code)


def test_lesion_table_counts_images_per_lesion(metadata):
    table = build_lesion_table(metadata)
    assert table["n_images"].sum() == len(metadata)
    assert table["lesion_id"].is_unique
    assert len(table) < len(metadata), "fixture must contain multi-image lesions"


def test_lesion_with_conflicting_diagnoses_rejected(metadata):
    corrupted = metadata.copy()
    lesion_id = corrupted["lesion_id"].value_counts().idxmax()
    first_row = corrupted.index[corrupted["lesion_id"] == lesion_id][0]
    original_dx = metadata.loc[first_row, "dx"]
    corrupted.loc[first_row, "dx"] = "mel" if original_dx == "vasc" else "vasc"

    with pytest.raises(ValueError, match="more than one diagnosis"):
        assign_splits(corrupted, RATIOS, seed=42)


def test_duplicate_image_ids_rejected(metadata):
    duplicated = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate image_ids"):
        assign_splits(duplicated, RATIOS, seed=42)


def test_missing_columns_rejected(metadata):
    with pytest.raises(ValueError, match="missing required columns"):
        assign_splits(metadata.drop(columns=["lesion_id"]), RATIOS, seed=42)
