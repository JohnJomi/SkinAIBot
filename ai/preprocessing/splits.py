"""Lesion-grouped, class-stratified train/val/test splitting.

HAM10000 contains roughly 10,015 images of only ~7,470 distinct lesions: the
same lesion is frequently photographed several times. Splitting at the image
level therefore puts near-duplicate views of one lesion on both sides of the
train/test boundary, and the resulting accuracy measures memorisation rather
than generalisation.

The algorithm below never splits a lesion, so overlap is impossible by
construction rather than by check:

1. Collapse the image table to one row per lesion (all images of a lesion share
   a diagnosis; a lesion that violates this raises).
2. Partition the lesions of each class independently. Because every class is
   split at the same ratios, the class distribution is preserved in all three
   splits - this is what provides stratification, without ever looking at an
   individual image.
3. Within a class, order lesions deterministically: sort by id, shuffle with the
   seeded RNG, then stable-sort by image count descending. Large lesions are
   placed first, while there is still slack to absorb them.
4. Assign each lesion to whichever split currently has the largest image
   deficit (target minus assigned), ties broken train > val > test.
5. Repair coverage: if a class has at least three lesions but some split got
   none, move the smallest lesion from the split holding the most.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ai.preprocessing.config import SplitRatios
from ai.preprocessing.labels import encode, validate_codes

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "lesion_id",
    "dx",
    "label_idx",
    "split",
)


def build_lesion_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Collapse the image-level metadata to one row per lesion.

    Returns columns `lesion_id`, `dx`, `n_images`. Raises if a lesion carries
    more than one diagnosis, which would make its class assignment ambiguous.
    """
    missing = {"lesion_id", "image_id", "dx"} - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")

    duplicate_images = metadata["image_id"][metadata["image_id"].duplicated()]
    if not duplicate_images.empty:
        raise ValueError(
            f"metadata contains duplicate image_ids: "
            f"{sorted(duplicate_images.unique())[:5]}"
        )

    validate_codes(metadata["dx"])

    grouped = metadata.groupby("lesion_id", sort=True).agg(
        dx=("dx", "unique"),
        n_images=("image_id", "size"),
    )

    mixed = grouped.index[grouped["dx"].map(len) > 1]
    if len(mixed) > 0:
        raise ValueError(
            f"lesions carry more than one diagnosis: {sorted(mixed)[:5]}; "
            "cannot assign them to a single class stratum"
        )

    grouped["dx"] = grouped["dx"].map(lambda values: values[0])
    return grouped.reset_index()[["lesion_id", "dx", "n_images"]]


def _order_lesions(lesions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Deterministic within-class ordering: shuffle, then largest lesions first.

    Sorting by lesion_id before shuffling makes the result depend only on the
    seed, not on the row order of the incoming metadata file.
    """
    ordered = lesions.sort_values("lesion_id", kind="mergesort").reset_index(drop=True)
    shuffled = ordered.iloc[rng.permutation(len(ordered))].reset_index(drop=True)
    # mergesort is stable, so the shuffled order breaks ties among equal sizes.
    return shuffled.sort_values(
        "n_images", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def _assign_class_lesions(
    lesions: pd.DataFrame,
    ratios: SplitRatios,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Partition one class's lesions across the three splits."""
    ordered = _order_lesions(lesions, rng)
    total_images = int(ordered["n_images"].sum())
    ratio_map = ratios.as_dict()
    targets = {name: ratio_map[name] * total_images for name in SPLIT_NAMES}

    assigned_images = {name: 0 for name in SPLIT_NAMES}
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    for lesion_id, n_images in zip(
        ordered["lesion_id"], ordered["n_images"], strict=True
    ):
        # max() keeps the first maximum, and SPLIT_NAMES is ordered
        # train > val > test, so ties resolve deterministically.
        target_split = max(
            SPLIT_NAMES, key=lambda name: targets[name] - assigned_images[name]
        )
        assigned[target_split].append(lesion_id)
        assigned_images[target_split] += int(n_images)

    return _repair_coverage(assigned, ordered)


def _repair_coverage(
    assigned: dict[str, list[str]], ordered: pd.DataFrame
) -> dict[str, list[str]]:
    """Ensure every split holds at least one lesion of this class.

    Only attempted when the class has enough lesions to go round; a class with
    two lesions genuinely cannot appear in three splits.
    """
    if len(ordered) < len(SPLIT_NAMES):
        return assigned

    sizes = dict(zip(ordered["lesion_id"], ordered["n_images"], strict=True))

    for name in SPLIT_NAMES:
        if assigned[name]:
            continue
        donor = max(
            SPLIT_NAMES,
            key=lambda candidate: (len(assigned[candidate]), candidate == "train"),
        )
        if len(assigned[donor]) < 2:
            continue
        # Move the donor's smallest lesion: least disruptive to the ratios.
        smallest = min(assigned[donor], key=lambda lesion: (sizes[lesion], lesion))
        assigned[donor].remove(smallest)
        assigned[name].append(smallest)

    return assigned


def assign_splits(
    metadata: pd.DataFrame,
    ratios: SplitRatios,
    seed: int,
) -> pd.DataFrame:
    """Produce the image-level split manifest.

    Returns one row per image with columns `image_id`, `lesion_id`, `dx`,
    `label_idx`, `split`. Every image of a given lesion receives the same
    split, so no lesion appears in more than one split.
    """
    lesion_table = build_lesion_table(metadata)
    rng = np.random.default_rng(seed)

    lesion_split: dict[str, str] = {}
    # Iterate classes in canonical order so the RNG is consumed identically on
    # every run regardless of how the metadata happens to be sorted.
    for dx in sorted(lesion_table["dx"].unique()):
        class_lesions = lesion_table[lesion_table["dx"] == dx]
        for split_name, lesion_ids in _assign_class_lesions(
            class_lesions, ratios, rng
        ).items():
            for lesion_id in lesion_ids:
                lesion_split[lesion_id] = split_name

    manifest = metadata[["image_id", "lesion_id", "dx"]].copy()
    manifest["label_idx"] = manifest["dx"].map(encode)
    manifest["split"] = manifest["lesion_id"].map(lesion_split)

    if manifest["split"].isna().any():
        raise RuntimeError("internal error: some images were left unassigned")

    return manifest.sort_values("image_id", kind="mergesort").reset_index(drop=True)[
        list(MANIFEST_COLUMNS)
    ]


def split_frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slice a manifest into its three per-split frames."""
    return {
        name: manifest[manifest["split"] == name].reset_index(drop=True)
        for name in SPLIT_NAMES
    }


def lesion_overlap(frames: Sequence[pd.DataFrame]) -> set[str]:
    """Return lesion ids appearing in more than one of `frames`.

    Used by `prepare` as a post-write assertion and by the test suite.
    """
    seen: set[str] = set()
    overlap: set[str] = set()
    for frame in frames:
        lesions = set(frame["lesion_id"])
        overlap |= seen & lesions
        seen |= lesions
    return overlap
