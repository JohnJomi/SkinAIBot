"""CLI: evaluate a selected checkpoint on the held-out test split.

    python -m ai.training.evaluate
        [--data-config PATH]      dataset config; default ai/configs/dataset.yaml
        [--training-config PATH]  training config; default ai/configs/training.yaml
        [--checkpoint PATH]       default: best.pt in the configured
                                  checkpoint_dir

This is the only module that reads test images. It runs once, after training
has chosen a checkpoint on validation, and it neither trains nor tunes: the
test split informs the report and nothing else.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from torch import nn

from ai.preprocessing.config import DEFAULT_CONFIG_PATH as DEFAULT_DATA_CONFIG_PATH
from ai.preprocessing.config import DataConfig, load_config
from ai.preprocessing.datamodule import build_dataloaders
from ai.preprocessing.labels import CLASS_CODES, CLASS_NAMES
from ai.training.checkpoints import BEST_CHECKPOINT_NAME, load_checkpoint
from ai.training.config import DEFAULT_CONFIG_PATH as DEFAULT_TRAINING_CONFIG_PATH
from ai.training.config import TrainingConfig, load_training_config
from ai.training.engine import evaluate as run_evaluation
from ai.training.engine import resolve_device
from ai.training.metrics import compute_metrics
from ai.training.model import build_model

REPORT_NAME = "report.json"
CONFUSION_MATRIX_NAME = "confusion_matrix.csv"


def evaluate_checkpoint(
    data_config: DataConfig,
    training_config: TrainingConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Score `checkpoint_path` on the test split and write the report."""
    device = resolve_device(training_config.device)

    # Weights come from the checkpoint, so never re-download pretrained ones.
    model = build_model(training_config.model.model_copy(update={"pretrained": False}))
    checkpoint = load_checkpoint(checkpoint_path, model=model)
    model.to(device)

    loaders = build_dataloaders(data_config)
    _, y_true, y_prob = run_evaluation(
        model, loaders.test, nn.CrossEntropyLoss(), device
    )

    metrics = compute_metrics(y_true, y_prob)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_stage": checkpoint["stage_name"],
        "selection_metric": checkpoint["metric_name"],
        "selection_metric_value": checkpoint["metric_value"],
        "model_name": checkpoint["model_name"],
        "split": "test",
        "n_images": int(len(y_true)),
        "class_codes": list(CLASS_CODES),
        "metrics": metrics,
    }

    _write_report(training_config.report_dir, report, metrics["confusion_matrix"])
    _print_summary(metrics, report)
    return report


def _write_report(
    report_dir: Path, report: dict[str, Any], confusion_matrix: list[list[int]]
) -> None:
    """Write report.json and confusion_matrix.csv into `report_dir`."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # NaN is not valid JSON, and `default=` is never consulted for floats, so
    # the undefined AUCs have to be replaced before encoding. They become null,
    # which stays distinguishable from a real 0.0; allow_nan=False then makes
    # any NaN we missed an error rather than invalid JSON.
    (report_dir / REPORT_NAME).write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    matrix_path = report_dir / CONFUSION_MATRIX_NAME
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *CLASS_CODES])
        for code, row in zip(CLASS_CODES, confusion_matrix, strict=True):
            writer.writerow([code, *row])


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN floats with None so the report is valid JSON.

    An undefined AUC must survive to the file as null; encoding it as 0.0 would
    read as "perfectly wrong" rather than "not measurable on this split".
    """
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _print_summary(metrics: dict[str, Any], report: dict[str, Any]) -> None:
    """Print the headline metrics and the per-class table to stdout.

    Undefined AUCs print as "undefined" rather than a number, so a class that
    could not be scored is never mistaken for one that scored badly.
    """
    print(f"\nTest split: {report['n_images']} images")
    print(f"  accuracy      {metrics['accuracy']:.4f}")
    print(f"  macro F1      {metrics['macro_f1']:.4f}")
    print(f"  macro recall  {metrics['macro_recall']:.4f}")
    print(f"  macro AUC     {metrics['macro_auc']:.4f}")

    header = f"\n{'class':>6}{'recall':>10}{'f1':>10}{'auc':>10}{'support':>10}"
    print(header)
    for code in CLASS_CODES:
        stats = metrics["per_class"][code]
        auc = f"{stats['auc']:.4f}" if stats["auc_defined"] else "undefined"
        print(
            f"{code:>6}{stats['recall']:>10.4f}{stats['f1']:>10.4f}"
            f"{auc:>10}{stats['support']:>10}   {CLASS_NAMES[code]}"
        )


def main() -> None:
    """CLI entry point: score a checkpoint, defaulting to best.pt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG_PATH)
    parser.add_argument(
        "--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to best.pt in the configured checkpoint directory",
    )
    args = parser.parse_args()

    training_config = load_training_config(args.training_config)
    checkpoint = args.checkpoint or (
        training_config.checkpoint_dir / BEST_CHECKPOINT_NAME
    )
    evaluate_checkpoint(load_config(args.data_config), training_config, checkpoint)


if __name__ == "__main__":
    main()
