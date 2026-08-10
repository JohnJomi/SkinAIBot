"""Evaluation metrics for Phase 2.2.

The arithmetic is scikit-learn's; this module owns two things it will not do
for us:

1. Every call fixes `labels=range(NUM_CLASSES)`, so a class missing from a
   batch cannot shift the column order of the confusion matrix or reorder the
   per-class arrays.
2. Per-class AUC is computed one class at a time and guarded. A one-shot
   `roc_auc_score(multi_class="ovr")` raises as soon as any class has no
   positives, taking the whole report down; and reporting 0.0 for an undefined
   AUC would read as "perfectly wrong" rather than "not measurable here".
   Undefined AUC is NaN, and the macro is the mean over the defined ones.
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES

LABEL_INDICES: list[int] = list(range(NUM_CLASSES))


def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to a proper probability distribution per row."""
    return torch.softmax(logits, dim=1)


def top_k_predictions(
    probabilities: np.ndarray | torch.Tensor, k: int = 3
) -> list[list[tuple[str, float]]]:
    """Per sample, the k most likely classes as (class_code, probability).

    A pure function over probabilities - Phase 2.3 can serve Top-3 from it
    without this phase owning any inference endpoint.
    """
    if not 1 <= k <= NUM_CLASSES:
        raise ValueError(f"k must be in [1, {NUM_CLASSES}], got {k}")

    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"expected probabilities of shape (n, {NUM_CLASSES}), "
            f"got {probabilities.shape}"
        )

    # argsort ascending, then reverse: highest probability first.
    ordered = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    return [
        [(CLASS_CODES[index], float(row[index])) for index in indices]
        for row, indices in zip(probabilities, ordered, strict=True)
    ]


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """One-vs-rest AUC per class; NaN where it is undefined.

    A class needs both positive and negative examples in the evaluated set for
    its ROC curve to exist. Anything else is NaN, never 0.0.
    """
    y_true = np.asarray(y_true)
    aucs = np.full(NUM_CLASSES, np.nan, dtype=float)

    for class_index in LABEL_INDICES:
        positives = np.asarray(y_true == class_index)
        n_positive = int(positives.sum())
        if n_positive == 0 or n_positive == len(y_true):
            continue
        aucs[class_index] = float(
            roc_auc_score(positives.astype(int), y_prob[:, class_index])
        )
    return aucs


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is among the k most likely.

    k=1 is ordinary accuracy; Top-3 is what the product surfaces alongside a
    prediction, so it is measured rather than assumed.
    """
    if not 1 <= k <= NUM_CLASSES:
        raise ValueError(f"k must be in [1, {NUM_CLASSES}], got {k}")

    y_true = np.asarray(y_true)
    top = np.argsort(np.asarray(y_prob), axis=1)[:, ::-1][:, :k]
    return float((top == y_true[:, None]).any(axis=1).mean())


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> list[dict[str, Any]]:
    """Confidence-versus-accuracy bins over the predicted class.

    An empty bin reports NaN for its mean confidence and accuracy rather than
    0.0: no samples means no measurement, not perfect miscalibration.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    confidence = y_prob.max(axis=1)
    correct = y_prob.argmax(axis=1) == y_true

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        # Half-open below, closed at the top, so 1.0 lands in the last bin.
        in_bin = (confidence > lower) & (confidence <= upper)
        if index == 0:
            in_bin |= confidence == lower
        count = int(in_bin.sum())
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": (
                    float(confidence[in_bin].mean()) if count else float("nan")
                ),
                "accuracy": float(correct[in_bin].mean()) if count else float("nan"),
            }
        )
    return bins


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Support-weighted mean gap between confidence and accuracy.

    0 means reported confidence matches observed accuracy. Confidence is a
    user-facing number here, so a model that is accurate but overconfident is
    a real defect and needs its own measurement.
    """
    bins = reliability_bins(y_true, y_prob, n_bins)
    total = sum(bin_["count"] for bin_ in bins)
    if total == 0:
        raise ValueError("cannot compute calibration error over an empty set")

    return float(
        sum(
            bin_["count"] / total * abs(bin_["accuracy"] - bin_["mean_confidence"])
            for bin_ in bins
            if bin_["count"]
        )
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k: int = 3,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    """Full metric set over a split's true labels and predicted probabilities.

    Recall is reported per class as well as macro-averaged: in this setting a
    missed malignant lesion is the expensive error, and a macro average can
    hide one class collapsing.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics over an empty evaluation set")
    # Rejected, not clamped: silently reporting Top-7 when Top-9 was asked for
    # would disagree with `top_k_predictions`, which raises on the same input.
    if not 1 <= top_k <= NUM_CLASSES:
        raise ValueError(f"top_k must be in [1, {NUM_CLASSES}], got {top_k}")

    y_pred = y_prob.argmax(axis=1)

    per_class_f1 = f1_score(
        y_true, y_pred, labels=LABEL_INDICES, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=LABEL_INDICES, average=None, zero_division=0
    )
    aucs = per_class_auc(y_true, y_prob)
    defined = ~np.isnan(aucs)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=LABEL_INDICES, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=LABEL_INDICES, average="macro", zero_division=0
            )
        ),
        # Support-weighted, the form docs/ROADMAP.md sets a target against.
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_INDICES,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=LABEL_INDICES,
                average="weighted",
                zero_division=0,
            )
        ),
        "top_k_accuracy": {
            str(k): top_k_accuracy(y_true, y_prob, k) for k in sorted({1, top_k})
        },
        "expected_calibration_error": expected_calibration_error(
            y_true, y_prob, calibration_bins
        ),
        # Mean over the classes whose AUC is defined; NaN if none are.
        "macro_auc": float(aucs[defined].mean()) if defined.any() else float("nan"),
        "per_class": {
            code: {
                "f1": float(per_class_f1[index]),
                "recall": float(per_class_recall[index]),
                "auc": float(aucs[index]),
                "auc_defined": bool(defined[index]),
                "support": int((y_true == index).sum()),
            }
            for index, code in enumerate(CLASS_CODES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABEL_INDICES
        ).tolist(),
    }
