"""The CLI emits clean JSON, refuses bad input, and never guesses."""

import json
import sys

import numpy as np
import pytest
import torch
import yaml

from ai.inference.predict import EXIT_ERROR, EXIT_OK, EXIT_REFUSED, main
from ai.preprocessing.dataset import index_images, load_manifest
from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.tests.test_predictor import b4_checkpoint  # noqa: F401 - fixture
from ai.training.thresholds import ThresholdSet, manifest_fingerprint

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def images(prepared_config):
    manifest = load_manifest(prepared_config.manifest_path("test"))
    index = index_images(prepared_config.raw_image_dirs)
    return [str(index[image_id]) for image_id in manifest["image_id"]][:2]


@pytest.fixture
def invoke(monkeypatch, config_path, b4_checkpoint):  # noqa: F811
    """Run `main()` with the given arguments; return (exit code, out, err)."""

    def run(*args, capsys, checkpoint=None, data_config=None):
        argv = [
            "predict",
            *args,
            "--checkpoint",
            str(checkpoint if checkpoint is not None else b4_checkpoint),
            "--data-config",
            str(data_config if data_config is not None else config_path),
            "--device",
            "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exit_info:
            main()
        captured = capsys.readouterr()
        code = exit_info.value.code
        return (0 if code is None else code), captured.out, captured.err

    return run


def _threshold_file(path, checkpoint_fingerprint, manifest_fingerprint_value, value,
                    fitted_on="val"):
    ThresholdSet(
        fitted_on=fitted_on,
        manifest_fingerprint=manifest_fingerprint_value,
        checkpoint_fingerprint=checkpoint_fingerprint,
        precision_floor=0.5,
        thresholds=np.full(NUM_CLASSES, value),
        precision=np.full(NUM_CLASSES, 1.0),
        recall=np.full(NUM_CLASSES, 1.0),
        feasible=np.ones(NUM_CLASSES, dtype=bool),
    ).save(path)
    return path


@pytest.fixture
def config_without_manifests(config_path, tmp_path):
    """A config YAML pointing at a manifest directory that does not exist."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["manifest_dir"] = str(tmp_path / "absent")
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# --- happy path -----------------------------------------------------------


def test_valid_image_exits_zero_with_json_on_stdout(invoke, images, capsys):
    code, out, _ = invoke(images[0], capsys=capsys)

    assert code == EXIT_OK
    payload = json.loads(out)
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["predicted_code"] in CLASS_CODES
    assert 0.0 <= result["confidence"] <= 1.0
    assert list(result["probabilities"]) == list(CLASS_CODES)
    assert len(result["top_k"]) == 3


def test_summary_goes_to_stderr_leaving_stdout_pure_json(invoke, images, capsys):
    _, out, err = invoke(images[0], capsys=capsys)

    # stdout must parse whole; nothing human-readable may leak into it.
    json.loads(out)
    assert "device:" in err
    assert "confidence" in err


def test_output_file_holds_the_same_result(invoke, images, tmp_path, capsys):
    destination = tmp_path / "nested" / "result.json"
    code, out, _ = invoke(images[0], "--output", str(destination), capsys=capsys)

    assert code == EXIT_OK
    assert out == ""
    assert json.loads(destination.read_text(encoding="utf-8"))["results"]


def test_unwritable_output_path_is_an_execution_error(
    invoke, images, tmp_path, capsys
):
    # The destination is a directory, so writing the file fails. That is an
    # execution failure, not a refused prediction, and must not traceback.
    destination = tmp_path / "already-a-directory"
    destination.mkdir()

    code, _, err = invoke(images[0], "--output", str(destination), capsys=capsys)

    assert code == EXIT_ERROR
    assert "failed to write output" in err
    assert "Traceback" not in err


def test_unserialisable_payload_is_an_execution_error(
    invoke, images, monkeypatch, capsys
):
    # allow_nan=False makes a non-finite probability a hard error rather than
    # invalid JSON; it must surface through the CLI error path.
    def nan_payload(*args, **kwargs):
        return {
            "device": "cpu",
            "thresholds": None,
            "results": [{"source": "x", "confidence": float("nan")}],
        }

    monkeypatch.setattr("ai.inference.predict.run", nan_payload)

    code, _, err = invoke(images[0], capsys=capsys)

    assert code == EXIT_ERROR
    assert "failed to write output" in err
    assert "Traceback" not in err


def test_multiple_images_preserve_input_order(invoke, images, capsys):
    code, out, _ = invoke(*images, capsys=capsys)

    assert code == EXIT_OK
    assert [r["source"] for r in json.loads(out)["results"]] == list(images)


def test_provenance_is_reported_for_every_image(invoke, images, capsys):
    _, out, _ = invoke(*images, capsys=capsys)

    for result in json.loads(out)["results"]:
        provenance = result["provenance"]
        assert provenance["architecture"] == "tf_efficientnet_b4"
        assert len(provenance["checkpoint_fingerprint"]) == 64
        assert provenance["class_codes"] == list(CLASS_CODES)


# --- refusals -------------------------------------------------------------


def test_missing_image_is_refused(invoke, tmp_path, capsys):
    code, out, err = invoke(str(tmp_path / "absent.jpg"), capsys=capsys)

    assert code == EXIT_REFUSED
    assert out == ""
    assert "refused" in err


def test_corrupt_image_is_refused(invoke, tmp_path, capsys):
    path = tmp_path / "broken.jpg"
    path.write_text("not an image", encoding="utf-8")

    code, _, err = invoke(str(path), capsys=capsys)

    assert code == EXIT_REFUSED
    assert "not a recognised image format" in err


def test_missing_checkpoint_is_refused(invoke, images, tmp_path, capsys):
    code, _, err = invoke(
        images[0], capsys=capsys, checkpoint=tmp_path / "absent.pt"
    )

    assert code == EXIT_REFUSED
    assert "checkpoint" in err


def test_invalid_checkpoint_is_refused(invoke, images, tmp_path, capsys):
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"not a torch checkpoint")

    code, _, err = invoke(images[0], capsys=capsys, checkpoint=path)

    # A malformed file is an execution failure, not an input we can classify.
    assert code in (EXIT_REFUSED, EXIT_ERROR)
    assert err.strip()


@pytest.mark.parametrize("top_k", ["0", str(NUM_CLASSES + 1)])
def test_invalid_top_k_is_refused(invoke, images, top_k, capsys):
    code, _, err = invoke(images[0], "--top-k", top_k, capsys=capsys)

    assert code == EXIT_REFUSED
    assert "--top-k must be in" in err


# --- thresholds -----------------------------------------------------------


def test_without_the_flag_there_is_no_threshold_block(invoke, images, capsys):
    _, out, _ = invoke(images[0], capsys=capsys)

    result = json.loads(out)["results"][0]
    assert result["threshold_prediction"] is None
    assert json.loads(out)["thresholds"] is None


def test_thresholds_are_never_discovered(invoke, images, monkeypatch, capsys):
    # Nothing may read a threshold file unless one was named on the command
    # line, however many are lying around.
    def forbidden(*args, **kwargs):
        raise AssertionError("a threshold file was loaded without --thresholds")

    monkeypatch.setattr("ai.inference.predict.ThresholdSet.load", forbidden)

    assert invoke(images[0], capsys=capsys)[0] == EXIT_OK


def test_explicit_thresholds_are_applied(
    invoke, images, tmp_path, prepared_config, b4_checkpoint, capsys  # noqa: F811
):
    from ai.training.thresholds import checkpoint_fingerprint

    path = _threshold_file(
        tmp_path / "thresholds.json",
        checkpoint_fingerprint(b4_checkpoint),
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.0,
    )

    code, out, _ = invoke(images[0], "--thresholds", str(path), capsys=capsys)

    assert code == EXIT_OK
    threshold = json.loads(out)["results"][0]["threshold_prediction"]
    assert threshold["abstained"] is False
    assert threshold["verification"]["manifest_verified"] is True
    assert threshold["verification"]["fitted_on"] == "val"


def test_abstention_keeps_the_argmax_result(
    invoke, images, tmp_path, prepared_config, b4_checkpoint, capsys  # noqa: F811
):
    from ai.training.thresholds import checkpoint_fingerprint

    path = _threshold_file(
        tmp_path / "thresholds.json",
        checkpoint_fingerprint(b4_checkpoint),
        manifest_fingerprint(prepared_config.manifest_path("val")),
        1.1,
    )

    code, out, err = invoke(images[0], "--thresholds", str(path), capsys=capsys)
    result = json.loads(out)["results"][0]

    assert code == EXIT_OK
    assert result["threshold_prediction"]["abstained"] is True
    assert result["threshold_prediction"]["predicted_code"] is None
    # The argmax answer is untouched.
    assert result["predicted_code"] in CLASS_CODES
    assert "abstained" in err


def test_threshold_set_from_another_checkpoint_is_refused(
    invoke, images, tmp_path, prepared_config, capsys
):
    path = _threshold_file(
        tmp_path / "thresholds.json",
        "f" * 64,
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.5,
    )

    code, _, err = invoke(images[0], "--thresholds", str(path), capsys=capsys)

    assert code == EXIT_REFUSED
    assert "different checkpoint" in err


def test_threshold_set_not_fitted_on_val_is_refused(
    invoke, images, tmp_path, prepared_config, b4_checkpoint, capsys  # noqa: F811
):
    from ai.training.thresholds import checkpoint_fingerprint

    path = _threshold_file(
        tmp_path / "thresholds.json",
        checkpoint_fingerprint(b4_checkpoint),
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.5,
        fitted_on="test",
    )

    code, _, err = invoke(images[0], "--thresholds", str(path), capsys=capsys)

    assert code == EXIT_REFUSED
    assert "only 'val'-fitted" in err


def test_stale_manifest_fingerprint_is_refused(
    invoke, images, tmp_path, b4_checkpoint, capsys  # noqa: F811
):
    from ai.training.thresholds import checkpoint_fingerprint

    path = _threshold_file(
        tmp_path / "thresholds.json",
        checkpoint_fingerprint(b4_checkpoint),
        "0" * 64,
        0.5,
    )

    code, _, err = invoke(images[0], "--thresholds", str(path), capsys=capsys)

    assert code == EXIT_REFUSED
    assert "different validation split" in err


def test_malformed_threshold_file_is_refused(invoke, images, tmp_path, capsys):
    path = tmp_path / "thresholds.json"
    path.write_text("{not json", encoding="utf-8")

    code, _, err = invoke(images[0], "--thresholds", str(path), capsys=capsys)

    assert code == EXIT_REFUSED
    assert "not valid JSON" in err


def test_unavailable_manifest_records_that_it_was_not_verified(
    invoke, images, tmp_path, prepared_config, b4_checkpoint,  # noqa: F811
    config_without_manifests, capsys
):
    from ai.training.thresholds import checkpoint_fingerprint

    expected = manifest_fingerprint(prepared_config.manifest_path("val"))
    path = _threshold_file(
        tmp_path / "thresholds.json",
        checkpoint_fingerprint(b4_checkpoint),
        expected,
        0.0,
    )

    code, out, err = invoke(
        images[0],
        "--thresholds",
        str(path),
        capsys=capsys,
        data_config=config_without_manifests,
    )

    assert code == EXIT_OK
    verification = json.loads(out)["results"][0]["threshold_prediction"][
        "verification"
    ]
    assert verification["manifest_verified"] is False
    assert verification["current_manifest_fingerprint"] is None
    assert verification["expected_manifest_fingerprint"] == expected
    assert "manifest_verified=false" in err


def test_unavailable_manifest_is_fatal_when_required(
    invoke, images, tmp_path, prepared_config, b4_checkpoint,  # noqa: F811
    config_without_manifests, capsys
):
    from ai.training.thresholds import checkpoint_fingerprint

    path = _threshold_file(
        tmp_path / "thresholds.json",
        checkpoint_fingerprint(b4_checkpoint),
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.0,
    )

    code, _, err = invoke(
        images[0],
        "--thresholds",
        str(path),
        "--require-manifest-check",
        capsys=capsys,
        data_config=config_without_manifests,
    )

    assert code == EXIT_REFUSED
    assert "is not available" in err
