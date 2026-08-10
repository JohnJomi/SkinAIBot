"""CLI: fine-tune EfficientNet-B4 across the configured stages.

    python -m ai.training.train [--data-config PATH] [--training-config PATH]

Trains on the training split and selects checkpoints on the validation split.

The test split is never read here. `build_dataloaders` constructs all three
loaders, and Phase 2.1 is not modified to change that, but constructing the
test Dataset only lists filenames and validates the manifest - no test image is
opened. This module binds `loaders.train` and `loaders.val` and never touches
`loaders.test`, so no test image is ever decoded during training. Test
evaluation happens once, in `ai.training.evaluate`, after model selection.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ai.preprocessing.config import DEFAULT_CONFIG_PATH as DEFAULT_DATA_CONFIG_PATH
from ai.preprocessing.config import DataConfig, load_config
from ai.preprocessing.datamodule import build_dataloaders
from ai.preprocessing.sampling import compute_class_weights
from ai.training.checkpoints import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    BestCheckpointTracker,
    save_checkpoint,
)
from ai.training.config import DEFAULT_CONFIG_PATH as DEFAULT_TRAINING_CONFIG_PATH
from ai.training.config import StageConfig, TrainingConfig, load_training_config
from ai.training.engine import evaluate, resolve_device, train_one_epoch
from ai.training.metrics import compute_metrics
from ai.training.model import (
    apply_stage,
    build_model,
    count_trainable_parameters,
    input_size_note,
    trainable_parameters,
)

HISTORY_NAME = "history.json"


def seed_everything(seed: int) -> None:
    """Seed every RNG the training run draws from.

    The seed comes from the dataset config, so the split and the training run
    are reproducible from one number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(device: torch.device) -> None:
    """Make CUDA kernel selection deterministic, where PyTorch supports it.

    Seeding alone does not make a CUDA run repeatable: cuDNN benchmarks several
    convolution algorithms and picks the fastest for the observed shapes, some
    kernels accumulate in nondeterministic order, and cuBLAS reuses workspaces
    across streams. Those are settings, not seeds.

    CPU and MPS are deliberately left untouched. `use_deterministic_algorithms`
    is global, and on CPU it turns ops that have no deterministic kernel into
    hard errors - a behaviour change on devices that were already reproducible
    here, since the DataLoader order and augmentation both derive from the
    seeded generators.

    Strict mode: an op with no deterministic implementation raises rather than
    warning. A warning scrolls past in a long run and leaves a checkpoint that
    cannot be reproduced but looks like it can - failing at the offending op is
    the honest outcome, and points straight at what to replace.
    """
    if device.type != "cuda":
        return

    # Required for deterministic cuBLAS; read when the cuBLAS handle is first
    # created, so it must be set before the first matmul. setdefault leaves an
    # operator-supplied value alone.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def build_criterion(
    training_config: TrainingConfig, train_labels: list[int]
) -> nn.CrossEntropyLoss:
    """Loss for the configured imbalance strategy.

    Only `weighted_loss` attaches class weights; `weighted_sampler` corrects
    the same imbalance in the sampler instead. The strategy is a single enum,
    so the two can never both be active.
    """
    weight = None
    if training_config.imbalance_strategy == "weighted_loss":
        weight = compute_class_weights(train_labels)
    return nn.CrossEntropyLoss(
        weight=weight, label_smoothing=training_config.label_smoothing
    )


def _json_safe_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace NaN metrics with None so history.json is valid JSON.

    macro_auc is NaN whenever no class had both outcomes in the validation
    split. Writing it as null matches the evaluation report; leaving it as NaN
    would emit a bare `NaN` literal that strict JSON parsers reject.
    """
    return [
        {
            key: (None if isinstance(value, float) and value != value else value)
            for key, value in entry.items()
        }
        for entry in history
    ]


def _build_scheduler(
    optimizer: torch.optim.Optimizer, stage: StageConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-stage LR schedule; None when the stage holds its rate fixed.

    Cosine annealing spans exactly the stage's epochs, so each stage completes
    its own decay rather than inheriting a partly-decayed rate.
    """
    if stage.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage.epochs)
    return None


def run_training(
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> list[dict[str, Any]]:
    """Run every configured stage; return the per-epoch history.

    Stages are iterated as data: adding one is a config edit, and nothing here
    refers to how many there are.
    """
    seed_everything(data_config.seed)
    device = resolve_device(training_config.device)
    configure_determinism(device)

    loaders = build_dataloaders(
        data_config,
        use_weighted_sampler=training_config.imbalance_strategy == "weighted_sampler",
    )
    # Deliberately bound one at a time: `loaders.test` is not referenced.
    train_loader = loaders.train
    val_loader = loaders.val

    model = build_model(training_config.model).to(device)
    note = input_size_note(model, data_config.image_size)
    if note:
        print(f"note: {note}")

    criterion = build_criterion(training_config, loaders.train_labels).to(device)
    tracker = BestCheckpointTracker(metric_name=training_config.monitor)
    config_snapshot = training_config.model_dump(mode="json")

    history: list[dict[str, Any]] = []
    epoch = 0

    for stage in training_config.stages:
        apply_stage(model, stage)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model), lr=stage.lr, weight_decay=stage.weight_decay
        )
        scheduler = _build_scheduler(optimizer, stage)

        print(
            f"\nstage {stage.name!r}: {stage.epochs} epochs, lr={stage.lr}, "
            f"unfreeze_blocks={stage.unfreeze_blocks}, "
            f"{count_trainable_parameters(model):,} trainable parameters"
        )

        for _ in range(stage.epochs):
            epoch += 1
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, stage
            )
            val_loss, y_true, y_prob = evaluate(model, val_loader, criterion, device)
            val_metrics = compute_metrics(y_true, y_prob)
            if scheduler is not None:
                scheduler.step()

            monitored = float(val_metrics[training_config.monitor])
            improved = tracker.update(monitored, epoch)

            history.append(
                {
                    "epoch": epoch,
                    "stage": stage.name,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_macro_auc": val_metrics["macro_auc"],
                    "monitored": monitored,
                    "is_best": improved,
                }
            )
            print(
                f"  epoch {epoch:>3}  train_loss {train_loss:.4f}  "
                f"val_loss {val_loss:.4f}  {training_config.monitor} "
                f"{monitored:.4f}{'  *best' if improved else ''}"
            )

            checkpoint_args = {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "stage_name": stage.name,
                "metric_name": training_config.monitor,
                "metric_value": monitored,
                "model_name": training_config.model.name,
                "config_snapshot": config_snapshot,
            }
            save_checkpoint(
                training_config.checkpoint_dir / LAST_CHECKPOINT_NAME,
                **checkpoint_args,
            )
            if improved:
                save_checkpoint(
                    training_config.checkpoint_dir / BEST_CHECKPOINT_NAME,
                    **checkpoint_args,
                )

    history_path = training_config.checkpoint_dir / HISTORY_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(_json_safe_history(history), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(
        f"\nbest {tracker.metric_name}={tracker.best_value:.4f} "
        f"at epoch {tracker.best_epoch}; wrote {history_path}"
    )
    return history


def main() -> None:
    """CLI entry point: load both configs and run every training stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG_PATH)
    parser.add_argument(
        "--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH
    )
    args = parser.parse_args()

    run_training(
        load_config(args.data_config), load_training_config(args.training_config)
    )


if __name__ == "__main__":
    main()
