"""Reporting and the acceptance gate for the evaluation layer.

Everything here is presentation and validation over numbers Part 1 already
computed - no metric is calculated in this module. Two ideas it does own:

- **Provenance.** A metric without the commit, split and manifest it came from
  is not reproducible evidence. Every report carries enough to identify the
  exact run that produced it.
- **The gate.** Comparing results to a documented target by eye does not scale
  and does not fail a build. `evaluate_targets` returns a verdict; only the CLI
  decides what to do about it.
"""

import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.preprocessing.labels import CLASS_CODES, CLASS_NAMES

# Bump when the shape of report.json changes in a way consumers would notice.
SCHEMA_VERSION = "1.0"

REPORT_MARKDOWN_NAME = "report.md"
PREDICTIONS_NAME = "predictions.csv"
RELIABILITY_NAME = "reliability.csv"

UNKNOWN_COMMIT = "unknown"


def git_commit(repo_root: Path | None = None) -> str:
    """The current commit, or "unknown" outside a working repository.

    Metadata must never be the reason an evaluation fails, so every failure
    mode here degrades to a string rather than raising.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_COMMIT
    return result.stdout.strip() or UNKNOWN_COMMIT


def build_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Schema version, commit and timestamp for a report."""
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(repo_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@dataclass(frozen=True)
class TargetResult:
    """One metric checked against its configured floor."""

    metric: str
    target: float
    value: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form; an undefined value becomes null, never 0.0."""
        return {
            "metric": self.metric,
            "target": self.target,
            "value": None if self.value != self.value else self.value,
            "passed": self.passed,
        }


def evaluate_targets(
    metrics: dict[str, Any], targets: dict[str, float]
) -> list[TargetResult]:
    """Check each configured target against the measured metrics.

    A metric that came back undefined (NaN) fails. Passing it would mean a
    model gets a clean bill of health precisely because it could not be
    measured, which is the opposite of what a gate is for.
    """
    results: list[TargetResult] = []
    for metric, target in sorted(targets.items()):
        if metric not in metrics:
            available = sorted(
                name for name, value in metrics.items() if isinstance(value, float)
            )
            raise KeyError(
                f"target names metric {metric!r}, which the evaluation does "
                f"not produce; available: {available}"
            )
        value = float(metrics[metric])
        # NaN fails every comparison, including this one - stated explicitly
        # so the intent is not mistaken for an oversight.
        passed = value == value and value >= target
        results.append(
            TargetResult(metric=metric, target=target, value=value, passed=passed)
        )
    return results


def targets_passed(results: list[TargetResult]) -> bool:
    """True when every checked target held. No targets is a pass."""
    return all(result.passed for result in results)


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    """One row per image: truth, prediction, confidence, Top-K and all probs.

    The header is sized to the widest Top-K present. Rows carrying fewer ranks
    get blank cells rather than raising, so one short row cannot cost the whole
    report - a ragged batch is a reason to look at the data, not to lose it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    top_k = max((len(row["top_k"]) for row in predictions), default=0)

    header = ["image_id", "true", "predicted", "correct", "confidence"]
    for rank in range(1, top_k + 1):
        header += [f"top{rank}_code", f"top{rank}_prob"]
    header += [f"p_{code}" for code in CLASS_CODES]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in predictions:
            record = [
                row["image_id"],
                row["true"],
                row["predicted"],
                row["correct"],
                f"{row['confidence']:.6f}",
            ]
            ranked = row["top_k"]
            for rank in range(top_k):
                if rank < len(ranked):
                    code, probability = ranked[rank]
                    record += [code, f"{probability:.6f}"]
                else:
                    record += ["", ""]
            record += [f"{row['probabilities'][code]:.6f}" for code in CLASS_CODES]
            writer.writerow(record)


def write_reliability_csv(path: Path, bins: list[dict[str, Any]]) -> None:
    """Calibration bins. Empty bins keep blank cells rather than a false 0.0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lower", "upper", "count", "mean_confidence", "accuracy"])
        for bin_ in bins:
            writer.writerow(
                [
                    f"{bin_['lower']:.4f}",
                    f"{bin_['upper']:.4f}",
                    bin_["count"],
                    _number(bin_["mean_confidence"]),
                    _number(bin_["accuracy"]),
                ]
            )


def _number(value: float, places: int = 4) -> str:
    """Format a float, rendering NaN as an empty cell."""
    return "" if value != value else f"{value:.{places}f}"


def _metric(value: Any, places: int = 4) -> str:
    """Format a metric for Markdown, naming undefined values explicitly."""
    if value is None:
        return "undefined"
    if isinstance(value, float) and value != value:
        return "undefined"
    return f"{float(value):.{places}f}"


def render_markdown(
    report: dict[str, Any],
    predictions: list[dict[str, Any]],
    worst_n: int = 10,
) -> str:
    """Render the human-readable report."""
    metrics = report["metrics"]
    lines = [
        f"# Evaluation report - {report['split']} split",
        "",
        f"**Model:** `{report['model_name']}`  ",
        f"**Checkpoint:** `{report['checkpoint']}` "
        f"(epoch {report['checkpoint_epoch']}, stage `{report['checkpoint_stage']}`)  ",
        f"**Selected on:** {report['selection_metric']} = "
        f"{_metric(report['selection_metric_value'])}  ",
        f"**Images:** {report['n_images']}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Accuracy | {_metric(metrics['accuracy'])} |",
        f"| Macro F1 | {_metric(metrics['macro_f1'])} |",
        f"| Weighted F1 | {_metric(metrics['weighted_f1'])} |",
        f"| Macro recall | {_metric(metrics['macro_recall'])} |",
        f"| Weighted recall | {_metric(metrics['weighted_recall'])} |",
        f"| Macro AUC | {_metric(metrics['macro_auc'])} |",
        f"| Expected calibration error | "
        f"{_metric(metrics['expected_calibration_error'])} |",
    ]
    for k, value in sorted(metrics.get("top_k_accuracy", {}).items()):
        lines.append(f"| Top-{k} accuracy | {_metric(value)} |")

    lines += ["", "## Per class", "", *_per_class_table(metrics)]
    lines += ["", "## Confusion matrix", "", *_confusion_table(metrics)]

    gate = report.get("gate")
    if gate:
        lines += ["", "## Acceptance gate", "", *_gate_table(gate)]

    operating_point = report.get("operating_point")
    if operating_point:
        lines += ["", "## Operating point", ""]
        lines += _operating_point_section(operating_point)

    lines += ["", "## Most confident mistakes", "", *_worst_table(predictions, worst_n)]
    lines += ["", "## Provenance", "", *_provenance_list(report)]
    return "\n".join(lines) + "\n"


def _per_class_table(metrics: dict[str, Any]) -> list[str]:
    rows = [
        "| Class | Name | Recall | F1 | AUC | Support |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for code in CLASS_CODES:
        stats = metrics["per_class"][code]
        auc = _metric(stats["auc"]) if stats["auc_defined"] else "undefined"
        rows.append(
            f"| `{code}` | {CLASS_NAMES[code]} | {_metric(stats['recall'])} | "
            f"{_metric(stats['f1'])} | {auc} | {stats['support']} |"
        )
    return rows


def _confusion_table(metrics: dict[str, Any]) -> list[str]:
    header = "| true \\ predicted | " + " | ".join(f"`{c}`" for c in CLASS_CODES) + " |"
    divider = "| --- " * (len(CLASS_CODES) + 1) + "|"
    rows = [header, divider]
    for code, row in zip(CLASS_CODES, metrics["confusion_matrix"], strict=True):
        rows.append(f"| `{code}` | " + " | ".join(str(value) for value in row) + " |")
    return rows


def _gate_table(gate: dict[str, Any]) -> list[str]:
    rows = ["| Metric | Target | Value | Result |", "| --- | --- | --- | --- |"]
    for result in gate["results"]:
        mark = "pass" if result["passed"] else "**FAIL**"
        rows.append(
            f"| {result['metric']} | {_metric(result['target'])} | "
            f"{_metric(result['value'])} | {mark} |"
        )
    rows += ["", f"**Overall:** {'pass' if gate['passed'] else '**FAIL**'}"]
    return rows


def _operating_point_section(operating_point: dict[str, Any]) -> list[str]:
    rows = [
        f"Fitted on `{operating_point['fitted_on']}` with precision floor "
        f"{_metric(operating_point['precision_floor'], 2)}; "
        f"coverage {_metric(operating_point['coverage'])}, "
        f"{operating_point['abstained']} abstained.",
        "",
        "| Class | Threshold | Recall | Precision | Support | Predicted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for code in CLASS_CODES:
        stats = operating_point["per_class"][code]
        threshold = _metric(stats["threshold"]) if stats["feasible"] else "infeasible"
        rows.append(
            f"| `{code}` | {threshold} | {_metric(stats['recall'])} | "
            f"{_metric(stats['precision'])} | {stats['support']} | "
            f"{stats['predicted']} |"
        )
    return rows


def _worst_table(predictions: list[dict[str, Any]], worst_n: int) -> list[str]:
    mistakes = [row for row in predictions if not row["correct"]]
    if not mistakes or worst_n == 0:
        return ["None." if not mistakes else "Not listed."]

    # Confidently wrong first: those are the failures worth looking at.
    mistakes.sort(key=lambda row: row["confidence"], reverse=True)
    rows = [
        "| Image | True | Predicted | Confidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in mistakes[:worst_n]:
        rows.append(
            f"| `{row['image_id']}` | `{row['true']}` | `{row['predicted']}` | "
            f"{_metric(row['confidence'])} |"
        )
    return rows


def _provenance_list(report: dict[str, Any]) -> list[str]:
    provenance = report.get("provenance", {})
    lines = [
        f"- Schema version: `{provenance.get('schema_version', 'unknown')}`",
        f"- Git commit: `{provenance.get('git_commit', UNKNOWN_COMMIT)}`",
        f"- Generated at: {provenance.get('generated_at', 'unknown')}",
        f"- Split: `{report['split']}`",
        f"- Manifest fingerprint: `{report['manifest_fingerprint']}`",
    ]
    operating_point = report.get("operating_point")
    if operating_point:
        lines.append(
            "- Threshold manifest fingerprint: "
            f"`{operating_point['threshold_manifest_fingerprint']}`"
        )
    return lines
