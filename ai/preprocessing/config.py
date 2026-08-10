"""Reproducible data configuration for the Phase 2.1 dataset pipeline.

Every value that changes what the pipeline produces lives in a YAML file rather
than in code, so a run is fully described by (config file, git commit).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ai/preprocessing/config.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ai" / "configs" / "dataset.yaml"


class SplitRatios(BaseModel):
    """Target proportion of images per split. Approximate: lesions are never
    divided, so the realised proportions land near, not exactly on, these."""

    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _must_sum_to_one(self) -> "SplitRatios":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


class NormalizationConfig(BaseModel):
    """Per-channel normalization statistics applied after tensor conversion."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @field_validator("std")
    @classmethod
    def _std_must_be_positive(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if any(component <= 0 for component in value):
            raise ValueError(f"normalization std must be positive, got {value}")
        return value


class DataConfig(BaseModel):
    """Full data configuration.

    `image_size` is the model-specific input resolution of the *current* Phase 2
    target (EfficientNet-B4, 380x380). It is read from config everywhere; no
    module hardcodes it, so retargeting another backbone is a config edit.
    """

    # numpy's SeedSequence rejects negative seeds, so catch it here rather than
    # part-way through `assign_splits`.
    seed: int = Field(ge=0)
    raw_image_dirs: tuple[Path, ...]
    metadata_csv: Path
    manifest_dir: Path
    image_size: int = Field(gt=0)
    resize_size: int = Field(gt=0)
    splits: SplitRatios
    normalization: NormalizationConfig
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)

    @model_validator(mode="after")
    def _resize_must_cover_crop(self) -> "DataConfig":
        if self.resize_size < self.image_size:
            raise ValueError(
                f"resize_size ({self.resize_size}) must be >= image_size "
                f"({self.image_size}); evaluation centre-crops after resizing"
            )
        return self

    def manifest_path(self, split: str) -> Path:
        """Path of the manifest CSV for one split."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split!r}")
        return self.manifest_dir / f"{split}.csv"


def _resolve(path: Path) -> Path:
    """Interpret relative config paths against the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path | str | None = None) -> DataConfig:
    """Load and validate the data configuration.

    Relative paths in the YAML are resolved against the repository root so the
    same config works regardless of the current working directory.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config = DataConfig.model_validate(raw)
    return config.model_copy(
        update={
            "raw_image_dirs": tuple(_resolve(d) for d in config.raw_image_dirs),
            "metadata_csv": _resolve(config.metadata_csv),
            "manifest_dir": _resolve(config.manifest_dir),
        }
    )
