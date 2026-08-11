"""Inference agrees with evaluation, end to end.

The headline test here is `test_inference_matches_evaluation_probabilities`.
Inference and evaluation are separate code paths over the same model, and the
one way they can silently disagree is preprocessing: a different resize, crop
or normalisation would leave both halves passing their own unit tests while
every number the engine produced was quietly wrong. Comparing them on the same
images is what makes that impossible.
"""

import numpy as np
import pytest
import torch
from torch import nn

from ai.preprocessing.datamodule import build_dataloaders
from ai.preprocessing.dataset import index_images, load_manifest
from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.tests.test_predictor import b4_checkpoint, predictor  # noqa: F401 - fixtures
from ai.training.engine import evaluate as run_evaluation
from ai.training.thresholds import (
    ThresholdSet,
    apply_thresholds_to_probabilities,
    build_split_predictions,
    manifest_fingerprint,
)
from ai.training.thresholds import apply_thresholds as apply_to_split

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def test_split_paths(prepared_config):
    """Every test-split image path, in manifest order."""
    manifest = load_manifest(prepared_config.manifest_path("test"))
    index = index_images(prepared_config.raw_image_dirs)
    return [index[image_id] for image_id in manifest["image_id"]]


def test_inference_matches_evaluation_probabilities(
    predictor, prepared_config, test_split_paths  # noqa: F811
):
    """The anti-drift test: same model, same images, same numbers."""
    loaders = build_dataloaders(prepared_config)

    # The predictor's own model, so preprocessing is the only variable left.
    _, _, evaluation_probabilities = run_evaluation(
        predictor._model, loaders.test, nn.CrossEntropyLoss(), predictor.device
    )

    predictions = predictor.predict_batch(list(test_split_paths))
    inference_probabilities = np.array(
        [[p.probabilities[code] for code in CLASS_CODES] for p in predictions]
    )

    assert inference_probabilities.shape == evaluation_probabilities.shape
    np.testing.assert_allclose(
        inference_probabilities, evaluation_probabilities, atol=1e-5
    )


def test_class_ordering_matches_evaluation(predictor, test_split_paths):  # noqa: F811
    prediction = predictor.predict(test_split_paths[0])

    # The evaluation arrays are indexed by CLASS_CODES position; the inference
    # dict must use exactly that order or the two could not be compared above.
    assert list(prediction.probabilities) == list(CLASS_CODES)
    assert prediction.provenance.class_codes == CLASS_CODES


def test_repeated_inference_is_bitwise_identical_on_one_device(
    predictor, test_split_paths  # noqa: F811
):
    first = predictor.predict_batch(test_split_paths[:2])
    second = predictor.predict_batch(test_split_paths[:2])

    for a, b in zip(first, second, strict=True):
        assert a.probabilities == b.probabilities
        assert a.confidence == b.confidence


def test_threshold_logic_is_shared_with_evaluation(
    predictor, prepared_config, test_split_paths  # noqa: F811
):
    """Evaluation and inference must resolve to identical threshold arithmetic."""
    loaders = build_dataloaders(prepared_config)
    _, y_true, y_prob = run_evaluation(
        predictor._model, loaders.test, nn.CrossEntropyLoss(), predictor.device
    )

    thresholds = ThresholdSet(
        fitted_on="val",
        manifest_fingerprint=manifest_fingerprint(
            prepared_config.manifest_path("val")
        ),
        checkpoint_fingerprint=predictor.checkpoint_fingerprint,
        precision_floor=0.5,
        thresholds=np.full(NUM_CLASSES, 0.1),
        precision=np.full(NUM_CLASSES, 1.0),
        recall=np.full(NUM_CLASSES, 1.0),
        feasible=np.ones(NUM_CLASSES, dtype=bool),
    )

    # The evaluation route, through the split-carrying type.
    split_predictions = build_split_predictions(
        loaders, "test", prepared_config, y_true, y_prob
    )
    via_evaluation = apply_to_split(split_predictions, thresholds)

    # The inference route, straight from the array.
    via_inference = apply_thresholds_to_probabilities(y_prob, thresholds)

    np.testing.assert_array_equal(via_evaluation, via_inference)


def test_end_to_end_threshold_decision_matches_the_shared_function(
    predictor, prepared_config, test_split_paths  # noqa: F811
):
    thresholds = ThresholdSet(
        fitted_on="val",
        manifest_fingerprint=manifest_fingerprint(
            prepared_config.manifest_path("val")
        ),
        checkpoint_fingerprint=predictor.checkpoint_fingerprint,
        precision_floor=0.5,
        thresholds=np.full(NUM_CLASSES, 0.15),
        precision=np.full(NUM_CLASSES, 1.0),
        recall=np.full(NUM_CLASSES, 1.0),
        feasible=np.ones(NUM_CLASSES, dtype=bool),
    )

    predictions = predictor.predict_batch(
        list(test_split_paths[:4]), thresholds=thresholds
    )
    probabilities = np.array(
        [[p.probabilities[code] for code in CLASS_CODES] for p in predictions]
    )
    expected = apply_thresholds_to_probabilities(probabilities, thresholds)

    for prediction, choice in zip(predictions, expected, strict=True):
        threshold_prediction = prediction.threshold_prediction
        if choice < 0:
            assert threshold_prediction.abstained is True
            assert threshold_prediction.predicted_code is None
        else:
            assert threshold_prediction.abstained is False
            assert threshold_prediction.predicted_code == CLASS_CODES[choice]
        # Argmax survives either way.
        assert prediction.predicted_code in CLASS_CODES
