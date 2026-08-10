"""Checkpoint writing, loading and best-epoch tracking.

Every checkpoint carries the label order it was trained under. Loading weights
against a different `CLASS_CODES` would not fail loudly - the tensor shapes
still match - it would just relabel every prediction, so the check is explicit.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ai.preprocessing.labels import CLASS_CODES

BEST_CHECKPOINT_NAME = "best.pt"
LAST_CHECKPOINT_NAME = "last.pt"


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage_name: str,
    metric_name: str,
    metric_value: float,
    model_name: str,
    config_snapshot: dict[str, Any],
) -> None:
    """Write a `.pt` checkpoint.

    The config snapshot is stored as JSON text rather than as nested objects so
    the file stays loadable under `weights_only=True`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "stage_name": stage_name,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "class_codes": list(CLASS_CODES),
            "model_name": model_name,
            "config_json": json.dumps(config_snapshot, default=str),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module | None = None) -> dict[str, Any]:
    """Load a checkpoint, refusing one trained under a different label order.

    Passing `model` also restores its weights.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint {path} not found; run `python -m ai.training.train` first"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    stored_codes = list(checkpoint.get("class_codes", []))
    if stored_codes != list(CLASS_CODES):
        raise ValueError(
            f"checkpoint {path} was trained with class order {stored_codes}, "
            f"but the current mapping is {list(CLASS_CODES)}; loading it would "
            "silently relabel every prediction"
        )

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@dataclass
class BestCheckpointTracker:
    """Tracks the best value seen for the monitored validation metric.

    All supported metrics are 'higher is better', so one comparison covers
    them. NaN never counts as an improvement.
    """

    metric_name: str
    best_value: float = float("-inf")
    best_epoch: int | None = None

    def update(self, value: float, epoch: int) -> bool:
        """Record `value`; return True if it is a new best."""
        if value != value:  # NaN
            return False
        if value <= self.best_value:
            return False
        self.best_value = float(value)
        self.best_epoch = epoch
        return True
