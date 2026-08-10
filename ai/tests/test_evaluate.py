"""Test-split evaluation writes a faithful report."""

import csv
import json
import sys

import numpy as np
import pytest
import torch

from ai.preprocessing.dataset import load_manifest
from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.tests.test_training_loop import TinyNet
from ai.training.checkpoints import save_checkpoint
from ai.training.config import (
    EvaluationTargets,
    ModelConfig,
    StageConfig,
    TrainingConfig,
)
from ai.training.evaluate import (
    CONFUSION_MATRIX_NAME,
    REPORT_NAME,
    THRESHOLDS_NAME,
    _write_report,
    evaluate_checkpoint,
)
from ai.training.evaluate import main as evaluate_main
from ai.training.report import (
    PREDICTIONS_NAME,
    RELIABILITY_NAME,
    REPORT_MARKDOWN_NAME,
    SCHEMA_VERSION,
)
from ai.training.thresholds import ThresholdSet


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


@pytest.fixture
def harness(prepared_config, tmp_path, monkeypatch):
    """A callable running `evaluate_checkpoint` on any split, plus its config."""
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

    def run(**kwargs):
        return evaluate_checkpoint(
            prepared_config, config, checkpoint_path, **kwargs
        )

    return run, config


def test_test_run_never_fits_thresholds(harness, monkeypatch):
    """The guarantee: no code path derives an operating point from test data."""
    run, _ = harness

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate fitted thresholds during a test-split run")

    monkeypatch.setattr("ai.training.evaluate.fit_thresholds", forbidden)

    result = run(split="test")
    assert result["split"] == "test"
    assert result["operating_point"] is None


def test_val_run_fits_and_freezes_an_operating_point(harness):
    run, config = harness

    result = run(split="val")

    assert result["operating_point"]["fitted_on"] == "val"
    saved = ThresholdSet.load(config.report_dir / THRESHOLDS_NAME)
    assert saved.fitted_on == "val"


def test_test_run_applies_only_explicitly_named_thresholds(harness, prepared_config):
    run, config = harness
    run(split="val")
    threshold_path = config.report_dir / THRESHOLDS_NAME
    fitted = ThresholdSet.load(threshold_path)

    # Present on disk but not requested: must not change the test numbers.
    assert run(split="test")["operating_point"] is None

    applied = run(split="test", thresholds_path=threshold_path)
    assert applied["operating_point"]["fitted_on"] == "val"
    # Frozen: the applied values are bitwise the ones validation produced.
    np.testing.assert_array_equal(
        np.array(
            [
                applied["operating_point"]["per_class"][code]["threshold"]
                for code in CLASS_CODES
            ],
            dtype=float,
        ),
        np.where(np.isnan(fitted.thresholds), np.nan, fitted.thresholds),
    )


def test_missing_threshold_file_is_fatal(harness, tmp_path):
    run, _ = harness

    with pytest.raises(FileNotFoundError, match="--split val"):
        run(split="test", thresholds_path=tmp_path / "absent.json")


def test_stale_threshold_fingerprint_is_rejected(harness, prepared_config):
    run, config = harness
    run(split="val")
    threshold_path = config.report_dir / THRESHOLDS_NAME

    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    payload["manifest_fingerprint"] = "0" * 64
    threshold_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different validation split"):
        run(split="test", thresholds_path=threshold_path)


def test_argmax_and_operating_point_metrics_stay_separate(harness):
    run, config = harness
    run(split="val")

    result = run(
        split="test", thresholds_path=config.report_dir / THRESHOLDS_NAME
    )

    # Threshold-derived numbers never leak into the argmax block.
    assert "coverage" not in result["metrics"]
    assert "abstained" not in result["metrics"]
    assert "accuracy" not in result["operating_point"]
    assert "confusion_matrix" not in result["operating_point"]


def test_per_image_predictions_align_with_the_manifest(harness, prepared_config):
    run, _ = harness
    result = run(split="test")

    manifest = load_manifest(prepared_config.manifest_path("test"))
    rows = result["predictions"]

    assert [row["image_id"] for row in rows] == list(manifest["image_id"])
    assert [row["true"] for row in rows] == [
        CLASS_CODES[index] for index in manifest["label_idx"]
    ]
    assert all(len(row["top_k"]) == 3 for row in rows)
    assert all(row["correct"] == (row["true"] == row["predicted"]) for row in rows)


def test_predictions_are_not_written_into_report_json(harness):
    run, config = harness
    result = run(split="test")

    payload = json.loads((config.report_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert "predictions" in result
    assert "predictions" not in payload


def test_evaluation_is_repeatable(harness):
    run, _ = harness

    first = run(split="test")
    second = run(split="test")

    assert first["metrics"] == second["metrics"]
    assert [row["confidence"] for row in first["predictions"]] == [
        row["confidence"] for row in second["predictions"]
    ]


def test_train_split_cannot_be_evaluated(harness):
    # The training loader is shuffled and augmented: neither aligned nor
    # deterministic, so it is not an evaluation split.
    run, _ = harness

    with pytest.raises(ValueError, match="split must be one of"):
        run(split="train")


def test_all_artifacts_are_written(harness):
    run, config = harness
    run(split="test")

    for name in (
        REPORT_NAME,
        REPORT_MARKDOWN_NAME,
        CONFUSION_MATRIX_NAME,
        PREDICTIONS_NAME,
        RELIABILITY_NAME,
    ):
        assert (config.report_dir / name).is_file(), name


def test_predictions_csv_covers_the_split(harness, prepared_config):
    run, config = harness
    result = run(split="test")

    with (config.report_dir / PREDICTIONS_NAME).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    manifest = load_manifest(prepared_config.manifest_path("test"))
    assert len(rows) == len(manifest)
    assert [row["image_id"] for row in rows] == list(manifest["image_id"])
    assert all(row["true"] in CLASS_CODES for row in rows)
    assert len(rows) == len(result["predictions"])


def test_report_json_carries_provenance(harness):
    run, config = harness
    result = run(split="test")

    payload = json.loads((config.report_dir / REPORT_NAME).read_text(encoding="utf-8"))
    provenance = payload["provenance"]

    assert provenance["schema_version"] == SCHEMA_VERSION
    assert provenance["git_commit"]
    assert provenance["generated_at"].endswith("+00:00")
    assert payload["manifest_fingerprint"] == result["manifest_fingerprint"]


def test_gate_result_is_recorded(harness):
    run, _ = harness
    result = run(split="test")

    # The stub model is untrained, so the shipped 0.85 target is not met here;
    # what matters is that the verdict is recorded rather than assumed.
    assert set(result["gate"]) == {"results", "passed"}
    assert isinstance(result["gate"]["passed"], bool)


def test_gate_passes_when_targets_are_met(harness, prepared_config, tmp_path):
    run, config = harness
    relaxed = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"targets": EvaluationTargets(accuracy=0.0)}
            )
        }
    )

    result = evaluate_checkpoint(
        prepared_config, relaxed, config.checkpoint_dir.parent / "best.pt", split="test"
    )

    assert result["gate"]["passed"] is True
    assert result["gate"]["results"][0]["metric"] == "accuracy"


def test_gate_fails_on_an_unreachable_target(harness, prepared_config):
    run, config = harness
    strict = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"targets": EvaluationTargets(accuracy=1.0)}
            )
        }
    )

    result = evaluate_checkpoint(
        prepared_config, strict, config.checkpoint_dir.parent / "best.pt", split="test"
    )

    assert result["gate"]["passed"] is False


def test_markdown_report_renders_the_split(harness):
    run, config = harness
    run(split="test")

    markdown = (config.report_dir / REPORT_MARKDOWN_NAME).read_text(encoding="utf-8")

    assert "# Evaluation report - test split" in markdown
    for code in CLASS_CODES:
        assert f"`{code}`" in markdown


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Drive `main()` with the evaluation itself stubbed out.

    The exit-code contract is the unit under test here, not the metrics.
    """

    def run(gate_passed: bool, argv: list[str] | None = None, boom: bool = False):
        monkeypatch.setattr(
            "ai.training.evaluate.load_training_config",
            lambda path: TrainingConfig(
                model=ModelConfig(name="stub", pretrained=False),
                stages=(
                    StageConfig(
                        name="head",
                        epochs=1,
                        lr=1e-3,
                        weight_decay=0.0,
                        unfreeze_blocks=0,
                    ),
                ),
                checkpoint_dir=tmp_path / "checkpoints",
                report_dir=tmp_path / "reports",
                device="cpu",
            ),
        )
        monkeypatch.setattr("ai.training.evaluate.load_config", lambda path: None)

        def fake_evaluate(*args, **kwargs):
            if boom:
                raise RuntimeError("checkpoint is corrupt")
            return {"gate": {"passed": gate_passed, "results": []}}

        monkeypatch.setattr(
            "ai.training.evaluate.evaluate_checkpoint", fake_evaluate
        )
        monkeypatch.setattr(sys, "argv", ["evaluate", *(argv or [])])

        with pytest.raises(SystemExit) as exit_info:
            evaluate_main()
            raise SystemExit(0)
        return exit_info.value.code

    return run


def test_cli_exits_zero_when_targets_are_met(cli):
    assert cli(gate_passed=True) in (0, None)


def test_cli_exits_one_when_a_target_is_missed(cli):
    assert cli(gate_passed=False) == 1


def test_cli_exits_two_when_the_run_fails(cli):
    # A failed run must be distinguishable from a completed run that missed.
    assert cli(gate_passed=True, boom=True) == 2


def test_cli_gate_can_be_disabled(cli):
    assert cli(gate_passed=False, argv=["--no-fail-under-targets"]) in (0, None)


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
