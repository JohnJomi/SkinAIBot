"""Torch Dataset over a committed split manifest.

The manifest stores `image_id`, not a file path, so it stays valid no matter
where the raw archives were extracted. Paths are resolved once at construction
against the configured image directories.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ai.preprocessing.labels import NUM_CLASSES

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def index_images(image_dirs: tuple[Path, ...] | list[Path]) -> dict[str, Path]:
    """Map `image_id` -> file path across all raw image directories."""
    index: dict[str, Path] = {}
    for directory in image_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(path.stem, path)
    return index


class HAM10000Dataset(Dataset):
    """Images of one split, paired with their class index.

    A dataset instance covers exactly one split; there is no mode flag that
    could be set wrongly and no way for an instance to reach another split's
    rows.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_dirs: tuple[Path, ...] | list[Path],
        transform: Callable[[Image.Image], Any],
    ) -> None:
        self.transform = transform
        self._index = index_images(image_dirs)

        self.image_ids: list[str] = list(manifest["image_id"])
        self.labels: list[int] = [int(value) for value in manifest["label_idx"]]

        missing = [
            image_id for image_id in self.image_ids if image_id not in self._index
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest images not found under {list(image_dirs)}; "
                f"first missing: {missing[:5]}"
            )

        out_of_range = [
            label for label in self.labels if not 0 <= label < NUM_CLASSES
        ]
        if out_of_range:
            raise ValueError(
                f"manifest contains label indices outside [0, {NUM_CLASSES}): "
                f"{sorted(set(out_of_range))}"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path = self._index[self.image_ids[index]]
        # HAM10000 is RGB JPEG, but converting defends against a stray
        # greyscale or RGBA file producing a wrong-channel tensor.
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), self.labels[index]


def load_manifest(path: Path) -> pd.DataFrame:
    """Read a split manifest CSV, keeping ids as strings."""
    if not path.is_file():
        raise FileNotFoundError(
            f"split manifest {path} not found; "
            "run `python -m ai.preprocessing.prepare` to generate it"
        )
    return pd.read_csv(path, dtype={"image_id": str, "lesion_id": str, "dx": str})
