"""Metrics match hand-computed values, and undefined AUC stays undefined."""

import numpy as np
import pytest
import torch

from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.training.metrics import (
    class_probabilities,
    compute_metrics,
    expected_calibration_error,
    per_class_auc,
    reliability_bins,
    top_k_accuracy,
    top_k_predictions,
)


@pytest.fixture
def two_class_case():
    """Four samples over classes 0 and 1; classes 2-6 never occur.

    Predictions are [0, 0, 1, 0] against truth [0, 0, 1, 1], so accuracy is
    3/4 and class 1's recall is 1/2 - small enough to verify by hand.
    """
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.zeros((4, NUM_CLASSES))
    y_prob[0, 0], y_prob[0, 1] = 0.9, 0.1
    y_prob[1, 0], y_prob[1, 1] = 0.8, 0.2
    y_prob[2, 0], y_prob[2, 1] = 0.3, 0.7
    y_prob[3, 0], y_prob[3, 1] = 0.6, 0.4
    return y_true, y_prob


def test_class_probabilities_normalise_logits():
    probabilities = class_probabilities(torch.randn(5, NUM_CLASSES))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)
    assert (probabilities >= 0).all()


def test_top_k_returns_codes_ranked_by_probability():
    probabilities = np.zeros((1, NUM_CLASSES))
    probabilities[0, 4] = 0.6
    probabilities[0, 1] = 0.3
    probabilities[0, 0] = 0.1

    (row,) = top_k_predictions(probabilities, k=3)
    assert [code for code, _ in row] == [
        CLASS_CODES[4],
        CLASS_CODES[1],
        CLASS_CODES[0],
    ]
    assert row[0][1] == pytest.approx(0.6)


def test_top_k_accepts_tensors_and_validates_its_arguments():
    probabilities = torch.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    assert len(top_k_predictions(probabilities, k=1)[0]) == 1

    with pytest.raises(ValueError, match="k must be in"):
        top_k_predictions(probabilities, k=NUM_CLASSES + 1)
    with pytest.raises(ValueError, match="expected probabilities of shape"):
        top_k_predictions(np.zeros((2, 3)))


def test_accuracy_and_recall_match_hand_computed_values(two_class_case):
    metrics = compute_metrics(*two_class_case)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["per_class"][CLASS_CODES[0]]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"][CLASS_CODES[1]]["recall"] == pytest.approx(0.5)
    # Macro averages over all seven classes; the five absent ones contribute 0.
    assert metrics["macro_recall"] == pytest.approx(1.5 / NUM_CLASSES)


def test_confusion_matrix_is_square_over_all_classes(two_class_case):
    matrix = np.array(compute_metrics(*two_class_case)["confusion_matrix"])

    assert matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert matrix.sum() == 4
    assert matrix[0].tolist() == [2, 0, 0, 0, 0, 0, 0]
    assert matrix[1].tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_support_counts_the_true_labels(two_class_case):
    per_class = compute_metrics(*two_class_case)["per_class"]

    assert per_class[CLASS_CODES[0]]["support"] == 2
    assert per_class[CLASS_CODES[1]]["support"] == 2
    assert per_class[CLASS_CODES[2]]["support"] == 0


def test_auc_is_nan_for_classes_without_both_outcomes(two_class_case):
    y_true, y_prob = two_class_case
    aucs = per_class_auc(y_true, y_prob)

    # Both present classes separate perfectly here.
    assert aucs[0] == pytest.approx(1.0)
    assert aucs[1] == pytest.approx(1.0)
    # Absent classes are undefined, not zero - 0.0 would read as "perfectly
    # wrong" and drag the macro down.
    assert np.isnan(aucs[2:]).all()
    assert not (aucs[2:] == 0.0).any()


def test_macro_auc_averages_only_the_defined_classes(two_class_case):
    metrics = compute_metrics(*two_class_case)

    assert metrics["macro_auc"] == pytest.approx(1.0)
    assert metrics["per_class"][CLASS_CODES[0]]["auc_defined"] is True
    assert metrics["per_class"][CLASS_CODES[2]]["auc_defined"] is False
    assert np.isnan(metrics["per_class"][CLASS_CODES[2]]["auc"])


def test_single_class_target_set_yields_all_undefined_auc():
    y_true = np.zeros(4, dtype=int)
    y_prob = np.full((4, NUM_CLASSES), 1.0 / NUM_CLASSES)

    metrics = compute_metrics(y_true, y_prob)

    assert np.isnan(per_class_auc(y_true, y_prob)).all()
    assert np.isnan(metrics["macro_auc"])
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_top_k_accuracy_brackets_are_exact(two_class_case):
    y_true, y_prob = two_class_case

    # k=1 is ordinary accuracy; k=NUM_CLASSES always contains the truth.
    assert top_k_accuracy(y_true, y_prob, 1) == pytest.approx(0.75)
    assert top_k_accuracy(y_true, y_prob, NUM_CLASSES) == pytest.approx(1.0)
    # Row 3 is wrong at k=1 but its true class is second, so k=2 recovers it.
    assert top_k_accuracy(y_true, y_prob, 2) == pytest.approx(1.0)


def test_top_k_accuracy_validates_k(two_class_case):
    y_true, y_prob = two_class_case

    with pytest.raises(ValueError, match="k must be in"):
        top_k_accuracy(y_true, y_prob, 0)
    with pytest.raises(ValueError, match="k must be in"):
        top_k_accuracy(y_true, y_prob, NUM_CLASSES + 1)


def test_weighted_f1_is_reported_alongside_macro(two_class_case):
    metrics = compute_metrics(*two_class_case)

    # Only classes 0 and 1 have support, so the weighted average ignores the
    # five empty classes that drag the macro down.
    assert metrics["weighted_f1"] > metrics["macro_f1"]
    assert metrics["weighted_recall"] == pytest.approx(0.75)


def test_top_k_accuracy_appears_in_the_metric_set(two_class_case):
    metrics = compute_metrics(*two_class_case, top_k=3)

    assert metrics["top_k_accuracy"]["1"] == pytest.approx(metrics["accuracy"])
    assert metrics["top_k_accuracy"]["3"] == pytest.approx(1.0)


def test_perfect_calibration_scores_zero():
    # Confidence 1.0 on every sample, and every sample correct.
    y_true = np.arange(NUM_CLASSES)
    y_prob = np.eye(NUM_CLASSES)

    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_overconfidence_is_detected():
    # 90% confident, 50% correct: a 0.4 gap the accuracy alone would not show.
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.full((4, NUM_CLASSES), 0.1 / (NUM_CLASSES - 1))
    y_prob[:, 0] = 0.9
    y_prob[2:, 0] = 0.05
    y_prob[2:, 1] = 0.9

    error = expected_calibration_error(y_true, y_prob)
    assert error == pytest.approx(0.4, abs=0.05)


def test_empty_reliability_bins_report_nan_not_zero():
    y_true = np.zeros(2, dtype=int)
    y_prob = np.zeros((2, NUM_CLASSES))
    y_prob[:, 0] = 1.0

    bins = reliability_bins(y_true, y_prob, n_bins=10)

    assert len(bins) == 10
    assert sum(bin_["count"] for bin_ in bins) == 2
    empty = [bin_ for bin_ in bins if bin_["count"] == 0]
    assert empty, "fixture must leave some bins empty"
    # No samples means no measurement, not perfect miscalibration.
    assert all(np.isnan(bin_["accuracy"]) for bin_ in empty)
    assert all(np.isnan(bin_["mean_confidence"]) for bin_ in empty)
    assert not any(bin_["accuracy"] == 0.0 for bin_ in empty)


def test_calibration_over_an_empty_set_rejected():
    with pytest.raises(ValueError, match="empty set"):
        expected_calibration_error(np.array([]), np.zeros((0, NUM_CLASSES)))


def test_empty_evaluation_set_rejected():
    with pytest.raises(ValueError, match="empty evaluation set"):
        compute_metrics(np.array([]), np.zeros((0, NUM_CLASSES)))
