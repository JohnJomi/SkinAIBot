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

    @model_validator(mode="after")
    def _pretrained_requires_an_explicit_tag(self) -> "ModelConfig":
        """A bare architecture name silently follows timm's default tag.

        Only enforced when actually loading pretrained weights: with
        `pretrained=False` nothing is downloaded, so the tag carries no meaning
        and a bare architecture name is fine.
        """
        if not self.pretrained:
            return self

        architecture, separator, tag = self.name.partition(".")
        if not (separator and architecture and tag):
            raise ValueError(
                f"model name {self.name!r} carries no explicit pretrained tag; "
                "use 'architecture.tag' (e.g. 'tf_efficientnet_b4.aa_in1k') so "
                "the weights cannot change when timm's default tag for the "
                "architecture does"
            )
        return self


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


class EvaluationTargets(BaseModel):
    """Minimum acceptable values for the evaluation gate.

    Every field is a floor - higher is better - so one comparison covers them
    all. Calibration error is deliberately absent: lower is better there, and
    silently comparing it the wrong way round would pass a badly calibrated
    model. Unset targets are not checked.
    """

    model_config = ConfigDict(extra="forbid")

    accuracy: float | None = Field(ge=0.0, le=1.0, default=None)
    macro_f1: float | None = Field(ge=0.0, le=1.0, default=None)
    weighted_f1: float | None = Field(ge=0.0, le=1.0, default=None)
    macro_recall: float | None = Field(ge=0.0, le=1.0, default=None)
    weighted_recall: float | None = Field(ge=0.0, le=1.0, default=None)
    macro_auc: float | None = Field(ge=0.0, le=1.0, default=None)

    def as_dict(self) -> dict[str, float]:
        """The targets that were actually set, by metric name."""
        return {
            name: value
            for name, value in self.model_dump().items()
            if value is not None
        }


class EvaluationConfig(BaseModel):
    """Settings for the evaluation and reporting layer."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(gt=0, default=3)
    calibration_bins: int = Field(gt=0, default=15)
    precision_floor: float = Field(ge=0.0, le=1.0, default=0.5)
    # Most-confident mistakes listed in the Markdown report.
    worst_n: int = Field(ge=0, default=10)
    targets: EvaluationTargets = Field(default_factory=EvaluationTargets)


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
    # Optional so configs written before the evaluation layer still load.
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _stage_names_must_be_unique(self) -> "TrainingConfig":
        """Stage names label checkpoints and history rows, so they must differ."""
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
        """Epochs across every stage - the length of a full training run."""
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
