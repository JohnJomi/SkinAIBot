"""Provenance, the acceptance gate, and the rendered artifacts."""

import csv
import re
import subprocess

import pytest

from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.training.report import (
    SCHEMA_VERSION,
    UNKNOWN_COMMIT,
    build_provenance,
    evaluate_targets,
    git_commit,
    render_markdown,
    targets_passed,
    write_predictions_csv,
    write_reliability_csv,
)

METRICS = {
    "accuracy": 0.80,
    "macro_f1": 0.60,
    "weighted_f1": 0.86,
    "macro_recall": 0.55,
    "weighted_recall": 0.80,
    "macro_auc": float("nan"),
    "expected_calibration_error": 0.12,
    "top_k_accuracy": {"1": 0.80, "3": 0.95},
    "per_class": {
        code: {
            "f1": 0.5,
            "recall": 0.5,
            "auc": 0.7 if index < 2 else float("nan"),
            "auc_defined": index < 2,
            "support": 10 if index < 2 else 0,
        }
        for index, code in enumerate(CLASS_CODES)
    },
    "confusion_matrix": [
        [1 if row == column else 0 for column in range(NUM_CLASSES)]
        for row in range(NUM_CLASSES)
    ],
}

PREDICTIONS = [
    {
        "image_id": "ISIC_0000001",
        "true": CLASS_CODES[0],
        "predicted": CLASS_CODES[1],
        "correct": False,
        "confidence": 0.97,
        "top_k": [
            (CLASS_CODES[1], 0.97),
            (CLASS_CODES[0], 0.02),
            (CLASS_CODES[2], 0.01),
        ],
        "probabilities": {code: 1.0 / NUM_CLASSES for code in CLASS_CODES},
    },
    {
        "image_id": "ISIC_0000002",
        "true": CLASS_CODES[1],
        "predicted": CLASS_CODES[1],
        "correct": True,
        "confidence": 0.60,
        "top_k": [
            (CLASS_CODES[1], 0.60),
            (CLASS_CODES[0], 0.30),
            (CLASS_CODES[2], 0.10),
        ],
        "probabilities": {code: 1.0 / NUM_CLASSES for code in CLASS_CODES},
    },
]


def _report(**overrides):
    report = {
        "checkpoint": "/tmp/best.pt",
        "checkpoint_epoch": 4,
        "checkpoint_stage": "finetune",
        "selection_metric": "macro_recall",
        "selection_metric_value": 0.55,
        "model_name": "tf_efficientnet_b4.aa_in1k",
        "split": "test",
        "manifest_fingerprint": "abc123",
        "n_images": 2,
        "class_codes": list(CLASS_CODES),
        "metrics": METRICS,
        "operating_point": None,
        "provenance": build_provenance(),
    }
    report.update(overrides)
    return report


# --- provenance -----------------------------------------------------------


def test_provenance_carries_schema_commit_and_time():
    provenance = build_provenance()

    assert provenance["schema_version"] == SCHEMA_VERSION
    assert provenance["git_commit"]
    # ISO-8601 with an explicit UTC offset.
    assert provenance["generated_at"].endswith("+00:00")


def test_git_commit_degrades_instead_of_raising(tmp_path, monkeypatch):
    # Metadata must never be the reason an evaluation fails.
    def explode(*args, **kwargs):
        raise OSError("git is not installed")

    monkeypatch.setattr(subprocess, "run", explode)
    assert git_commit(tmp_path) == UNKNOWN_COMMIT


def test_git_commit_outside_a_repository_is_unknown(tmp_path):
    assert git_commit(tmp_path) in {UNKNOWN_COMMIT} or len(git_commit(tmp_path)) == 40


# --- acceptance gate ------------------------------------------------------


def test_target_met_passes():
    results = evaluate_targets(METRICS, {"weighted_f1": 0.85})

    assert [result.metric for result in results] == ["weighted_f1"]
    assert results[0].passed is True
    assert targets_passed(results)


def test_target_missed_fails():
    results = evaluate_targets(METRICS, {"macro_recall": 0.90})

    assert results[0].passed is False
    assert not targets_passed(results)


def test_target_exactly_on_the_boundary_passes():
    assert evaluate_targets({"accuracy": 0.80}, {"accuracy": 0.80})[0].passed is True


def test_undefined_metric_fails_rather_than_passing_vacuously():
    # macro_auc is NaN here: a model must not be cleared because a metric
    # could not be measured.
    results = evaluate_targets(METRICS, {"macro_auc": 0.5})

    assert results[0].passed is False
    assert not targets_passed(results)
    # And it serialises as null, never 0.0.
    assert results[0].as_dict()["value"] is None


def test_no_targets_is_a_pass():
    assert targets_passed(evaluate_targets(METRICS, {})) is True


def test_target_naming_an_unknown_metric_is_rejected():
    with pytest.raises(KeyError, match="does not produce"):
        evaluate_targets(METRICS, {"f1_score": 0.5})


# --- markdown -------------------------------------------------------------


def test_markdown_lists_every_class():
    markdown = render_markdown(_report(), PREDICTIONS)

    for code in CLASS_CODES:
        assert f"`{code}`" in markdown


def test_markdown_names_undefined_values_rather_than_printing_zero():
    markdown = render_markdown(_report(), PREDICTIONS)

    assert "undefined" in markdown
    # The NaN macro AUC must not surface as a number.
    assert "| Macro AUC | undefined |" in markdown
    # No bare "nan" cell anywhere. Word-bounded, so "provenance" is not a hit.
    assert re.search(r"\bnan\b", markdown, re.IGNORECASE) is None


def test_markdown_includes_headline_sections():
    markdown = render_markdown(_report(), PREDICTIONS)

    for heading in (
        "## Headline metrics",
        "## Per class",
        "## Confusion matrix",
        "## Most confident mistakes",
        "## Provenance",
    ):
        assert heading in markdown


def test_markdown_reports_the_gate_verdict():
    results = evaluate_targets(METRICS, {"macro_recall": 0.90})
    gate = {
        "results": [result.as_dict() for result in results],
        "passed": targets_passed(results),
    }

    markdown = render_markdown(_report(gate=gate), PREDICTIONS)

    assert "## Acceptance gate" in markdown
    assert "**FAIL**" in markdown


def test_markdown_lists_confident_mistakes_first():
    markdown = render_markdown(_report(), PREDICTIONS)

    assert "ISIC_0000001" in markdown
    # The correct row is not a mistake and must not be listed.
    mistakes = markdown.split("## Most confident mistakes")[1]
    assert "ISIC_0000002" not in mistakes.split("## Provenance")[0]


def test_markdown_provenance_names_the_split_and_manifest():
    markdown = render_markdown(_report(), PREDICTIONS)

    assert f"Schema version: `{SCHEMA_VERSION}`" in markdown
    assert "Split: `test`" in markdown
    assert "Manifest fingerprint: `abc123`" in markdown


# --- csv artifacts --------------------------------------------------------


def test_predictions_csv_has_one_row_per_image(tmp_path):
    path = tmp_path / "predictions.csv"
    write_predictions_csv(path, PREDICTIONS)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(PREDICTIONS)
    assert rows[0]["image_id"] == "ISIC_0000001"
    assert rows[0]["true"] == CLASS_CODES[0]
    assert rows[0]["top1_code"] == CLASS_CODES[1]
    assert float(rows[0]["confidence"]) == pytest.approx(0.97)
    # Every class probability is present and named.
    for code in CLASS_CODES:
        assert f"p_{code}" in rows[0]


def test_predictions_csv_handles_rows_with_different_top_k_lengths(tmp_path):
    # A short row used to raise IndexError and cost the whole report.
    ragged = [
        PREDICTIONS[0],
        {
            **PREDICTIONS[1],
            "image_id": "ISIC_0000003",
            "top_k": [(CLASS_CODES[1], 0.60)],
        },
    ]
    path = tmp_path / "predictions.csv"
    write_predictions_csv(path, ragged)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Header is sized to the widest row.
    assert "top3_code" in rows[0]
    assert rows[0]["top3_code"] == CLASS_CODES[2]
    # The short row gets blanks, not a crash and not a fabricated value.
    assert rows[1]["top1_code"] == CLASS_CODES[1]
    assert rows[1]["top2_code"] == ""
    assert rows[1]["top2_prob"] == ""
    assert rows[1]["top3_code"] == ""


def test_reliability_csv_leaves_empty_bins_blank(tmp_path):
    bins = [
        {
            "lower": 0.0,
            "upper": 0.5,
            "count": 0,
            "mean_confidence": float("nan"),
            "accuracy": float("nan"),
        },
        {
            "lower": 0.5,
            "upper": 1.0,
            "count": 4,
            "mean_confidence": 0.75,
            "accuracy": 0.5,
        },
    ]
    path = tmp_path / "reliability.csv"
    write_reliability_csv(path, bins)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Blank, not 0.0: an empty bin was not measured.
    assert rows[0]["accuracy"] == ""
    assert rows[0]["mean_confidence"] == ""
    assert float(rows[1]["accuracy"]) == pytest.approx(0.5)
