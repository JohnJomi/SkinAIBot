"""CLI: turn the raw HAM10000 download into committed split manifests.

    python -m ai.preprocessing.prepare [--config PATH]

Run once after extracting the archives into `ai/data/raw/`. The manifests it
writes to `ai/data_manifests/` are committed; the images are not.
"""

import argparse
from pathlib import Path

import pandas as pd

from ai.preprocessing.config import DEFAULT_CONFIG_PATH, load_config
from ai.preprocessing.dataset import index_images
from ai.preprocessing.labels import CLASS_CODES
from ai.preprocessing.splits import (
    SPLIT_NAMES,
    assign_splits,
    lesion_overlap,
    split_frames,
)


def write_manifests(config_path: Path) -> dict[str, pd.DataFrame]:
    """Build and write the three split manifests. Returns them by split name."""
    config = load_config(config_path)

    metadata = pd.read_csv(
        config.metadata_csv, dtype={"image_id": str, "lesion_id": str, "dx": str}
    )
    manifest = assign_splits(metadata, config.splits, config.seed)

    available = index_images(config.raw_image_dirs)
    missing = sorted(set(manifest["image_id"]) - set(available))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images listed in the metadata are missing from "
            f"{list(config.raw_image_dirs)}; first missing: {missing[:5]}"
        )

    frames = split_frames(manifest)

    # The guarantee this whole pipeline exists to provide - assert it rather
    # than trust it, since a silent violation would inflate every later metric.
    overlap = lesion_overlap([frames[name] for name in SPLIT_NAMES])
    if overlap:
        raise RuntimeError(
            f"internal error: {len(overlap)} lesions span multiple splits"
        )

    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        frames[name].to_csv(config.manifest_path(name), index=False)

    return frames


def _report(frames: dict[str, pd.DataFrame]) -> None:
    total = sum(len(frame) for frame in frames.values())
    print(f"Wrote {total} images across {len(SPLIT_NAMES)} splits\n")
    header = f"{'class':>6}" + "".join(f"{name:>10}" for name in SPLIT_NAMES)
    print(header)
    for code in CLASS_CODES:
        counts = "".join(
            f"{int((frames[name]['dx'] == code).sum()):>10}" for name in SPLIT_NAMES
        )
        print(f"{code:>6}{counts}")
    totals = "".join(f"{len(frames[name]):>10}" for name in SPLIT_NAMES)
    print(f"{'total':>6}{totals}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to the data configuration YAML",
    )
    args = parser.parse_args()
    _report(write_manifests(args.config))


if __name__ == "__main__":
    main()
