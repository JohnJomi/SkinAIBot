"""Synthetic HAM10000-shaped fixture.

The real dataset is never downloaded or committed, so the suite generates a
miniature stand-in with the properties that matter: the published column names,
the seven class codes, several images sharing one lesion_id, and a heavy class
imbalance.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from ai.preprocessing.config import load_config
from ai.preprocessing.labels import CLASS_CODES

# Lesions per class, and images per lesion. Mirrors HAM10000's shape in
# miniature: nv dominates, df and vasc are rare, and most classes contain
# multi-image lesions so a leaking split has something to leak.
LESIONS_PER_CLASS: dict[str, list[int]] = {
    "akiec": [2, 1, 1, 1],
    "bcc": [3, 2, 1, 1, 1],
    "bkl": [2, 2, 1, 1, 1, 1],
    "df": [1, 1, 1],
    "mel": [3, 2, 2, 1, 1, 1],
    "nv": [4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    "vasc": [2, 1, 1],
}

# Not 600x450: the pipeline must not assume the source resolution.
SOURCE_IMAGE_SIZE = (160, 120)


def _build_metadata() -> pd.DataFrame:
    rows = []
    for dx, lesion_sizes in LESIONS_PER_CLASS.items():
        for lesion_number, n_images in enumerate(lesion_sizes):
            lesion_id = f"LES_{dx}_{lesion_number:03d}"
            for image_number in range(n_images):
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": f"ISIC_{dx}_{lesion_number:03d}_{image_number}",
                        "dx": dx,
                        "dx_type": "histo",
                        "age": 50.0,
                        "sex": "male",
                        "localization": "back",
                    }
                )
    # Shuffle so nothing can depend on the metadata arriving grouped by class.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


@pytest.fixture(scope="session")
def metadata() -> pd.DataFrame:
    """Image-level metadata with the published HAM10000 columns."""
    return _build_metadata()


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory, metadata: pd.DataFrame):
    """A directory tree shaped like an extracted HAM10000 download."""
    root = tmp_path_factory.mktemp("ham10000")
    raw = root / "raw"
    # Two image directories, as HAM10000 ships two archives.
    part_dirs = [raw / "part_1", raw / "part_2"]
    for directory in part_dirs:
        directory.mkdir(parents=True)

    rng = np.random.default_rng(0)
    for position, image_id in enumerate(metadata["image_id"]):
        pixels = rng.integers(
            0, 256, size=(*SOURCE_IMAGE_SIZE[::-1], 3), dtype=np.uint8
        )
        target = part_dirs[position % 2] / f"{image_id}.jpg"
        Image.fromarray(pixels).save(target, format="JPEG")

    metadata.to_csv(raw / "metadata.csv", index=False)
    return root


@pytest.fixture(scope="session")
def config_path(dataset_root, tmp_path_factory):
    """A data config pointing at the synthetic dataset, with absolute paths."""
    raw = dataset_root / "raw"
    payload = {
        "seed": 42,
        "raw_image_dirs": [str(raw / "part_1"), str(raw / "part_2")],
        "metadata_csv": str(raw / "metadata.csv"),
        "manifest_dir": str(dataset_root / "manifests"),
        # Small, and not the production 380: image_size must be read from
        # config rather than hardcoded anywhere in the pipeline.
        "image_size": 32,
        "resize_size": 36,
        "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "batch_size": 4,
        # Serial loading: worker processes would slow the suite for no gain.
        "num_workers": 0,
    }
    path = tmp_path_factory.mktemp("config") / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def config(config_path):
    return load_config(config_path)


@pytest.fixture(scope="session")
def prepared_config(config_path):
    """Config whose split manifests have been written to disk."""
    from ai.preprocessing.prepare import write_manifests

    write_manifests(config_path)
    return load_config(config_path)


@pytest.fixture(scope="session")
def all_class_codes() -> tuple[str, ...]:
    return CLASS_CODES
