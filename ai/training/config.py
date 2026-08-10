"""Reproducible training configuration for the Phase 2.2 fine-tuning pipeline.

Mirrors the Phase 2.1 approach: every value that changes what a run produces
lives in YAML, so a run is described by (dataset config, training config, git
commit). The seed is deliberately absent - it is read from the dataset config,
keeping one seed for the split and the training run.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai.preprocessing.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ai" / "configs" / "training.yaml"

ImbalanceStrategy = Literal["none", "weighted_sampler", "weighted_loss"]
MonitorMetric = Literal["accuracy", "macro_f1", "macro_recall"]
SchedulerName = Literal["cosine", "none"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]


class ModelConfig(BaseModel):
    """Which pretrained backbone to fine-tune.

    `name` is a full timm weight tag (architecture.pretrained_tag), not a bare
    architecture: timm's default tag for an architecture can change between
    releases, which would silently change the initialisation.
    """

    name: str
    pretrained: bool = True
    drop_rate: float = Field(ge=0.0, lt=1.0, default=0.4)


class StageConfig(BaseModel):
    """One fine-tuning stage.

    `unfreeze_blocks` counts backbone blocks from the end; 0 trains the
    classification head alone. It is validated against the model's real block
    count at stage-application time, not assumed here.
    """

    name: str
    epochs: int = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    unfreeze_blocks: int = Field(ge=0)
    scheduler: SchedulerName = "cosine"


class TrainingConfig(BaseModel):
    """Full training configuration."""

    # `model` is a field name here, not pydantic's own namespace.
    model_config = ConfigDict(protected_namespaces=())

    model: ModelConfig
    stages: tuple[StageConfig, ...] = Field(min_length=1)
    imbalance_strategy: ImbalanceStrategy = "weighted_loss"
    label_smoothing: float = Field(ge=0.0, lt=1.0, default=0.0)
    monitor: MonitorMetric = "macro_recall"
    checkpoint_dir: Path
    report_dir: Path
    device: DeviceName = "auto"

    @model_validator(mode="after")
    def _stage_names_must_be_unique(self) -> "TrainingConfig":
        names = [stage.name for stage in self.stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"stage names must be unique; repeated: {duplicates}. "
                "They label checkpoints and history entries."
            )
        return self

    @property
    def total_epochs(self) -> int:
        return sum(stage.epochs for stage in self.stages)


def _resolve(path: Path) -> Path:
    """Interpret relative config paths against the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    """Load and validate the training configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config = TrainingConfig.model_validate(raw)
    return config.model_copy(
        update={
            "checkpoint_dir": _resolve(config.checkpoint_dir),
            "report_dir": _resolve(config.report_dir),
        }
    )
