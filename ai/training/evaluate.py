"""CLI: evaluate a selected checkpoint on a held-out split.

    python -m ai.training.evaluate
        [--data-config PATH]      dataset config; default ai/configs/dataset.yaml
        [--training-config PATH]  training config; default ai/configs/training.yaml
        [--checkpoint PATH]       default: best.pt in the configured
                                  checkpoint_dir
        [--split {val,test}]      default: test
        [--top-k N]               Top-K accuracy to report; default 3
        [--precision-floor F]     precision floor for threshold fitting on val
        [--thresholds PATH]       apply a val-fitted operating point

This is the only module that reads test images. It runs after training has
chosen a checkpoint on validation, and it neither trains nor tunes on test.

Operating points are fitted on validation only, and never during a test run:
`fit_thresholds` is called on the `--split val` branch alone. A test run can
*apply* a frozen threshold set, but only when one is named explicitly with
`--thresholds`, and only after its recorded validation fingerprint is checked
against the validation manifest on disk.

Argmax metrics and threshold-based operating-point metrics are reported as
separate blocks; they answer different questions and must not be blended.
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
from ai.training.metrics import compute_metrics, reliability_bins, top_k_predictions
from ai.training.model import build_model
from ai.training.report import (
    PREDICTIONS_NAME,
    RELIABILITY_NAME,
    REPORT_MARKDOWN_NAME,
    build_provenance,
    evaluate_targets,
    render_markdown,
    targets_passed,
    write_predictions_csv,
    write_reliability_csv,
)
from ai.training.thresholds import (
    EVALUABLE_SPLITS,
    FITTING_SPLIT,
    ThresholdSet,
    build_split_predictions,
    fit_thresholds,
    manifest_fingerprint,
    operating_point_metrics,
)

REPORT_NAME = "report.json"
CONFUSION_MATRIX_NAME = "confusion_matrix.csv"
THRESHOLDS_NAME = "thresholds.json"

DEFAULT_PRECISION_FLOOR = 0.5


def evaluate_checkpoint(
    data_config: DataConfig,
    training_config: TrainingConfig,
    checkpoint_path: Path,
    split: str = "test",
    top_k: int | None = None,
    precision_floor: float | None = None,
    thresholds_path: Path | None = None,
) -> dict[str, Any]:
    """Score `checkpoint_path` on one split and write the report artifacts.

    Thresholds are fitted only when `split` is validation. On any other split
    an operating point can be applied but never derived, so the held-out data
    cannot feed back into how the model is used.

    `top_k` and `precision_floor` fall back to the training config's
    `evaluation` block when not given explicitly.
    """
    if split not in EVALUABLE_SPLITS:
        raise ValueError(
            f"split must be one of {list(EVALUABLE_SPLITS)}, got {split!r}"
        )

    settings = training_config.evaluation
    top_k = settings.top_k if top_k is None else top_k
    precision_floor = (
        settings.precision_floor if precision_floor is None else precision_floor
    )

    device = resolve_device(training_config.device)

    # Weights come from the checkpoint, so never re-download pretrained ones.
    model = build_model(training_config.model.model_copy(update={"pretrained": False}))
    checkpoint = load_checkpoint(checkpoint_path, model=model)
    model.to(device)

    loaders = build_dataloaders(data_config)
    # `split` selects the loader and becomes the provenance tag, so the two
    # cannot drift apart.
    _, y_true, y_prob = run_evaluation(
        model, getattr(loaders, split), nn.CrossEntropyLoss(), device
    )
    predictions = build_split_predictions(
        loaders, split, data_config, y_true, y_prob
    )

    metrics = compute_metrics(
        predictions.y_true,
        predictions.y_prob,
        top_k=top_k,
        calibration_bins=settings.calibration_bins,
    )

    thresholds = _resolve_thresholds(
        predictions, data_config, training_config, precision_floor, thresholds_path
    )

    target_results = evaluate_targets(metrics, settings.targets.as_dict())

    report = {
        "provenance": build_provenance(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_stage": checkpoint["stage_name"],
        "selection_metric": checkpoint["metric_name"],
        "selection_metric_value": checkpoint["metric_value"],
        "model_name": checkpoint["model_name"],
        "split": predictions.split,
        "manifest_fingerprint": predictions.manifest_fingerprint,
        "n_images": len(predictions),
        "class_codes": list(CLASS_CODES),
        # Argmax metrics, deliberately kept apart from the operating point.
        "metrics": metrics,
        "operating_point": (
            operating_point_metrics(predictions, thresholds)
            if thresholds is not None
            else None
        ),
        "gate": {
            "results": [result.as_dict() for result in target_results],
            "passed": targets_passed(target_results),
        },
    }

    # Per-image rows go to predictions.csv, never into report.json.
    rows = _per_image_predictions(predictions, top_k)
    bins = reliability_bins(
        predictions.y_true, predictions.y_prob, settings.calibration_bins
    )

    _write_report(training_config.report_dir, report, metrics["confusion_matrix"])
    _write_artifacts(training_config.report_dir, report, rows, bins, settings.worst_n)
    _print_summary(metrics, report)

    report["predictions"] = rows
    report["reliability"] = bins
    return report


def _write_artifacts(
    report_dir: Path,
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    worst_n: int,
) -> None:
    """Write the human-readable report and the per-row CSVs."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / REPORT_MARKDOWN_NAME).write_text(
        render_markdown(report, predictions, worst_n), encoding="utf-8"
    )
    write_predictions_csv(report_dir / PREDICTIONS_NAME, predictions)
    write_reliability_csv(report_dir / RELIABILITY_NAME, bins)


def _resolve_thresholds(
    predictions: Any,
    data_config: DataConfig,
    training_config: TrainingConfig,
    precision_floor: float,
    thresholds_path: Path | None,
) -> ThresholdSet | None:
    """Fit an operating point on validation, or load a frozen one.

    The two branches are exclusive by construction: fitting happens only on the
    validation branch, so no code path can derive thresholds from test data.
    """
    if predictions.split == FITTING_SPLIT and thresholds_path is None:
        thresholds = fit_thresholds(predictions, precision_floor=precision_floor)
        thresholds.save(training_config.report_dir / THRESHOLDS_NAME)
        return thresholds

    if thresholds_path is None:
        return None

    # Explicit request: a missing or unreadable file is fatal. Falling back to
    # fitting here is exactly the mistake the split guard exists to prevent.
    thresholds = ThresholdSet.load(thresholds_path)
    _verify_threshold_provenance(thresholds, data_config)
    return thresholds


def _verify_threshold_provenance(
    thresholds: ThresholdSet, data_config: DataConfig
) -> None:
    """Fail loudly if the thresholds no longer match the validation manifest."""
    current = manifest_fingerprint(data_config.manifest_path(FITTING_SPLIT))
    if thresholds.manifest_fingerprint != current:
        raise ValueError(
            "threshold set was fitted against a different validation split "
            f"(fingerprint {thresholds.manifest_fingerprint[:12]}..., current "
            f"{current[:12]}...); refit with `--split {FITTING_SPLIT}`"
        )


def _per_image_predictions(predictions: Any, top_k: int) -> list[dict[str, Any]]:
    """One row per image: truth, prediction, confidence and Top-K."""
    probabilities = predictions.y_prob
    ranked = top_k_predictions(probabilities, k=top_k)

    rows: list[dict[str, Any]] = []
    for index, image_id in enumerate(predictions.image_ids):
        true_index = int(predictions.y_true[index])
        predicted_index = int(probabilities[index].argmax())
        rows.append(
            {
                "image_id": image_id,
                "true": CLASS_CODES[true_index],
                "predicted": CLASS_CODES[predicted_index],
                "correct": true_index == predicted_index,
                "confidence": float(probabilities[index][predicted_index]),
                "top_k": ranked[index],
                "probabilities": {
                    code: float(probabilities[index][position])
                    for position, code in enumerate(CLASS_CODES)
                },
            }
        )
    return rows


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

    gate = report.get("gate")
    if gate and gate["results"]:
        print("\nAcceptance gate")
        for result in gate["results"]:
            value = "undefined" if result["value"] is None else f"{result['value']:.4f}"
            outcome = "pass" if result["passed"] else "FAIL"
            print(
                f"  {result['metric']:<18} target {result['target']:.4f}  "
                f"actual {value:>9}  {outcome}"
            )
        print(f"  overall: {'pass' if gate['passed'] else 'FAIL'}")


def main() -> None:
    """CLI entry point: score a checkpoint, defaulting to best.pt.

    Exit codes: 0 every target met, 1 a target missed, 2 the run failed.
    """
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
    parser.add_argument(
        "--split",
        choices=EVALUABLE_SPLITS,
        default="test",
        help="split to evaluate; thresholds are only ever fitted on val",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--precision-floor",
        type=float,
        default=DEFAULT_PRECISION_FLOOR,
        help=f"precision floor when fitting thresholds on {FITTING_SPLIT}",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="apply a val-fitted operating point from this file",
    )
    parser.add_argument(
        "--fail-under-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exit non-zero when a configured target is missed",
    )
    args = parser.parse_args()

    try:
        training_config = load_training_config(args.training_config)
        checkpoint = args.checkpoint or (
            training_config.checkpoint_dir / BEST_CHECKPOINT_NAME
        )
        report = evaluate_checkpoint(
            load_config(args.data_config),
            training_config,
            checkpoint,
            split=args.split,
            top_k=args.top_k,
            precision_floor=args.precision_floor,
            thresholds_path=args.thresholds,
        )
    except Exception as error:  # noqa: BLE001 - the CLI boundary
        # A failed run is distinct from a run that completed and missed a
        # target; CI needs to tell those apart.
        print(f"evaluation failed: {error}")
        raise SystemExit(2) from error

    if args.fail_under_targets and not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
