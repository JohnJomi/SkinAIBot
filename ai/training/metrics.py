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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Full metric set over a split's true labels and predicted probabilities.

    Recall is reported per class as well as macro-averaged: in this setting a
    missed malignant lesion is the expensive error, and a macro average can
    hide one class collapsing.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics over an empty evaluation set")

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
