"""The inference engine loads a validated checkpoint and predicts from it."""

import numpy as np
import pytest
import torch
from torch import nn

from ai.inference.predictor import Predictor, validate_architecture
from ai.preprocessing.dataset import index_images, load_manifest
from ai.preprocessing.labels import CLASS_CODES, CLASS_NAMES, NUM_CLASSES
from ai.training.checkpoints import save_checkpoint
from ai.training.config import ModelConfig
from ai.training.model import build_model
from ai.training.thresholds import ThresholdSet, checkpoint_fingerprint

# EfficientNet dispatches many small ops; multi-threaded CPU execution thrashes
# on them. See ai/tests/test_model.py for the measurement.
torch.set_num_threads(1)

MODEL_NAME = "tf_efficientnet_b4.aa_in1k"


@pytest.fixture(scope="session")
def b4_checkpoint(tmp_path_factory):
    """A real EfficientNet-B4 checkpoint with random weights.

    The architecture is the thing under test, so this is the genuine model
    rather than a stub; the fixture config uses 32px inputs, which keeps a
    forward pass well under a second.
    """
    model = build_model(ModelConfig(name=MODEL_NAME, pretrained=False))
    path = tmp_path_factory.mktemp("checkpoints") / "best.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        epoch=7,
        stage_name="finetune",
        metric_name="macro_recall",
        metric_value=0.42,
        model_name=MODEL_NAME,
        config_snapshot={"monitor": "macro_recall"},
    )
    return path


@pytest.fixture(scope="session")
def predictor(b4_checkpoint, prepared_config):
    return Predictor(b4_checkpoint, prepared_config, device="cpu")


@pytest.fixture(scope="session")
def sample_images(prepared_config):
    """Real image paths from the synthetic test split, in manifest order."""
    manifest = load_manifest(prepared_config.manifest_path("test"))
    index = index_images(prepared_config.raw_image_dirs)
    return [index[image_id] for image_id in manifest["image_id"]][:3]


def _threshold_set(fingerprint, manifest_fingerprint_value, value, fitted_on="val"):
    """A ThresholdSet with every class at `value`."""
    return ThresholdSet(
        fitted_on=fitted_on,
        manifest_fingerprint=manifest_fingerprint_value,
        checkpoint_fingerprint=fingerprint,
        precision_floor=0.5,
        thresholds=np.full(NUM_CLASSES, value),
        precision=np.full(NUM_CLASSES, 1.0),
        recall=np.full(NUM_CLASSES, 1.0),
        feasible=np.ones(NUM_CLASSES, dtype=bool),
    )


# --- architecture validation ---------------------------------------------


def test_supported_architecture_accepted():
    assert validate_architecture(MODEL_NAME) == "tf_efficientnet_b4"
    assert validate_architecture("efficientnet_b4.ra2_in1k") == "efficientnet_b4"


@pytest.mark.parametrize(
    "name", ["resnet50.a1_in1k", "tf_efficientnet_b0.aa_in1k", "stub", ""]
)
def test_unsupported_architecture_rejected(name):
    # Otherwise checkpoint metadata would decide which network gets built.
    with pytest.raises(ValueError, match="not a supported|does not record"):
        validate_architecture(name)


def test_checkpoint_naming_another_architecture_is_refused(
    tmp_path, prepared_config
):
    model = nn.Linear(4, NUM_CLASSES)
    path = tmp_path / "wrong.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        epoch=1,
        stage_name="head",
        metric_name="macro_recall",
        metric_value=0.1,
        model_name="resnet50.a1_in1k",
        config_snapshot={},
    )

    with pytest.raises(ValueError, match="not a supported"):
        Predictor(path, prepared_config, device="cpu")


def test_checkpoint_with_a_different_class_order_is_refused(
    b4_checkpoint, prepared_config, monkeypatch
):
    monkeypatch.setattr(
        "ai.training.checkpoints.CLASS_CODES", ("vasc", *CLASS_CODES[1:])
    )

    with pytest.raises(ValueError, match="silently relabel every prediction"):
        Predictor(b4_checkpoint, prepared_config, device="cpu")


def test_missing_checkpoint_reports_how_to_produce_one(tmp_path, prepared_config):
    with pytest.raises(FileNotFoundError, match="ai.training.train"):
        Predictor(tmp_path / "absent.pt", prepared_config, device="cpu")


# --- prediction shape and ordering ---------------------------------------


def test_model_is_in_eval_mode(predictor):
    assert predictor._model.training is False


def test_prediction_probabilities_are_a_distribution(predictor, sample_images):
    prediction = predictor.predict(sample_images[0])

    assert len(prediction.probabilities) == NUM_CLASSES
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(value >= 0 for value in prediction.probabilities.values())


def test_probability_keys_follow_the_canonical_class_order(predictor, sample_images):
    prediction = predictor.predict(sample_images[0])

    assert list(prediction.probabilities) == list(CLASS_CODES)


def test_argmax_matches_the_highest_probability(predictor, sample_images):
    prediction = predictor.predict(sample_images[0])

    best = max(prediction.probabilities.items(), key=lambda item: item[1])
    assert prediction.predicted_code == best[0]
    assert prediction.confidence == pytest.approx(best[1])
    assert prediction.predicted_name == CLASS_NAMES[prediction.predicted_code]


def test_top_k_is_ordered_and_sized(predictor, sample_images):
    prediction = predictor.predict(sample_images[0], top_k=3)

    probabilities = [entry.probability for entry in prediction.top_k]
    assert len(prediction.top_k) == 3
    assert probabilities == sorted(probabilities, reverse=True)
    assert prediction.top_k[0].code == prediction.predicted_code


@pytest.mark.parametrize("top_k", [0, NUM_CLASSES + 1])
def test_invalid_top_k_rejected(predictor, sample_images, top_k):
    with pytest.raises(ValueError, match="top_k must be in"):
        predictor.predict(sample_images[0], top_k=top_k)


def test_no_gradients_are_built(predictor, sample_images):
    # torch.inference_mode: tensors carry no grad history at all.
    with torch.enable_grad():
        predictor.predict(sample_images[0])
    assert all(not p.grad_fn for p in predictor._model.parameters())
    assert all(p.grad is None for p in predictor._model.parameters())


def test_repeated_inference_is_bitwise_identical(predictor, sample_images):
    first = predictor.predict(sample_images[0])
    second = predictor.predict(sample_images[0])

    assert first.probabilities == second.probabilities
    assert first.confidence == second.confidence


def test_batch_preserves_input_order(predictor, sample_images):
    batch = predictor.predict_batch(list(sample_images))
    individually = [predictor.predict(path) for path in sample_images]

    assert [p.source for p in batch] == [str(path) for path in sample_images]
    for batched, single in zip(batch, individually, strict=True):
        assert batched.probabilities == pytest.approx(single.probabilities, abs=1e-5)


def test_empty_batch_returns_nothing(predictor):
    assert predictor.predict_batch([]) == []


def test_provenance_comes_from_the_checkpoint(predictor, b4_checkpoint):
    provenance = predictor.provenance

    assert provenance.model_name == MODEL_NAME
    assert provenance.architecture == "tf_efficientnet_b4"
    assert provenance.epoch == 7
    assert provenance.stage_name == "finetune"
    assert provenance.selection_metric == "macro_recall"
    assert provenance.class_codes == CLASS_CODES
    assert provenance.checkpoint_fingerprint == checkpoint_fingerprint(b4_checkpoint)


# --- thresholds -----------------------------------------------------------


def test_no_thresholds_means_no_threshold_block(predictor, sample_images):
    assert predictor.predict(sample_images[0]).threshold_prediction is None


def test_thresholds_are_applied_when_supplied(
    predictor, sample_images, prepared_config
):
    from ai.training.thresholds import manifest_fingerprint

    thresholds = _threshold_set(
        predictor.checkpoint_fingerprint,
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.0,
    )
    prediction = predictor.predict(sample_images[0], thresholds=thresholds)

    assert prediction.threshold_prediction is not None
    assert prediction.threshold_prediction.abstained is False
    # Every threshold is 0.0, so the margin ranking reduces to the argmax.
    assert prediction.threshold_prediction.predicted_code == prediction.predicted_code


def test_abstention_keeps_the_argmax_prediction(
    predictor, sample_images, prepared_config
):
    from ai.training.thresholds import manifest_fingerprint

    # No probability can reach 1.1, so nothing qualifies.
    thresholds = _threshold_set(
        predictor.checkpoint_fingerprint,
        manifest_fingerprint(prepared_config.manifest_path("val")),
        1.1,
    )
    prediction = predictor.predict(sample_images[0], thresholds=thresholds)

    assert prediction.threshold_prediction.abstained is True
    assert prediction.threshold_prediction.predicted_code is None
    assert prediction.threshold_prediction.predicted_name is None
    # The argmax answer is still there: abstention adds information, it does
    # not remove any.
    assert prediction.predicted_code in CLASS_CODES
    assert prediction.confidence > 0


def test_thresholds_from_another_checkpoint_are_refused(
    predictor, sample_images, prepared_config
):
    from ai.training.thresholds import manifest_fingerprint

    thresholds = _threshold_set(
        "f" * 64,
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.5,
    )

    with pytest.raises(ValueError, match="different checkpoint"):
        predictor.predict(sample_images[0], thresholds=thresholds)


def test_thresholds_not_fitted_on_val_are_refused(
    predictor, sample_images, prepared_config
):
    from ai.training.thresholds import manifest_fingerprint

    thresholds = _threshold_set(
        predictor.checkpoint_fingerprint,
        manifest_fingerprint(prepared_config.manifest_path("val")),
        0.5,
        fitted_on="test",
    )

    with pytest.raises(ValueError, match="only 'val'-fitted"):
        predictor.predict(sample_images[0], thresholds=thresholds)
