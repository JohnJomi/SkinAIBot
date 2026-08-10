"""Test-split evaluation writes a faithful report."""

import csv
import json

import pytest
import torch

from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.tests.test_training_loop import TinyNet
from ai.training.checkpoints import save_checkpoint
from ai.training.config import ModelConfig, StageConfig, TrainingConfig
from ai.training.evaluate import (
    CONFUSION_MATRIX_NAME,
    REPORT_NAME,
    _write_report,
    evaluate_checkpoint,
)


@pytest.fixture
def report(prepared_config, tmp_path, monkeypatch):
    """Evaluate an untrained stub checkpoint against the synthetic test split."""
    model = TinyNet()
    monkeypatch.setattr("ai.training.evaluate.build_model", lambda config: model)

    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        epoch=2,
        stage_name="finetune",
        metric_name="macro_recall",
        metric_value=0.31,
        model_name="stub",
        config_snapshot={},
    )

    config = TrainingConfig(
        model=ModelConfig(name="stub", pretrained=False),
        stages=(
            StageConfig(
                name="head", epochs=1, lr=1e-3, weight_decay=0.0, unfreeze_blocks=0
            ),
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        device="cpu",
    )
    return evaluate_checkpoint(prepared_config, config, checkpoint_path), config


def test_report_describes_the_checkpoint_it_scored(report):
    result, _ = report

    assert result["split"] == "test"
    assert result["checkpoint_epoch"] == 2
    assert result["selection_metric"] == "macro_recall"
    assert result["class_codes"] == list(CLASS_CODES)
    assert result["n_images"] > 0


def test_report_files_are_written(report):
    _, config = report

    assert (config.report_dir / REPORT_NAME).is_file()
    assert (config.report_dir / CONFUSION_MATRIX_NAME).is_file()


def test_report_json_matches_the_returned_metrics(report):
    result, config = report
    payload = json.loads((config.report_dir / REPORT_NAME).read_text(encoding="utf-8"))

    assert payload["metrics"]["confusion_matrix"] == result["metrics"][
        "confusion_matrix"
    ]
    assert payload["n_images"] == result["n_images"]


def test_undefined_auc_is_written_as_null(tmp_path):
    # The synthetic test split contains every class, so no AUC is undefined
    # there. Drive the encoder directly: NaN is not valid JSON, and json.dumps
    # never consults `default=` for floats, so this path needs its own cover.
    report = {
        "metrics": {
            "macro_auc": float("nan"),
            "per_class": {
                "df": {"auc": float("nan"), "auc_defined": False, "recall": 0.0},
                "nv": {"auc": 0.75, "auc_defined": True, "recall": 1.0},
            },
        },
    }
    identity = [
        [1 if row == column else 0 for column in range(NUM_CLASSES)]
        for row in range(NUM_CLASSES)
    ]

    _write_report(tmp_path, report, identity)
    payload = json.loads((tmp_path / REPORT_NAME).read_text(encoding="utf-8"))

    assert payload["metrics"]["macro_auc"] is None
    assert payload["metrics"]["per_class"]["df"]["auc"] is None
    # A defined AUC is untouched, and null never becomes 0.0.
    assert payload["metrics"]["per_class"]["nv"]["auc"] == pytest.approx(0.75)
    assert payload["metrics"]["per_class"]["df"]["auc"] != 0.0


def test_confusion_matrix_csv_is_labelled_in_class_order(report):
    _, config = report
    with (config.report_dir / CONFUSION_MATRIX_NAME).open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["true\\predicted", *CLASS_CODES]
    assert [row[0] for row in rows[1:]] == list(CLASS_CODES)
    assert sum(int(value) for row in rows[1:] for value in row[1:]) > 0
