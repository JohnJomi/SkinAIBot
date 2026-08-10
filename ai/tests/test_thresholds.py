"""Thresholds are fitted on validation only, and frozen once fitted."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import SequentialSampler

from ai.preprocessing.datamodule import build_dataloaders
from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.training.thresholds import (
    FITTING_SPLIT,
    SplitPredictions,
    ThresholdSet,
    apply_thresholds,
    build_split_predictions,
    checkpoint_fingerprint,
    fit_thresholds,
    manifest_fingerprint,
    operating_point_metrics,
)

# Stand-in for a real checkpoint hash where the weights are not the subject.
CHECKPOINT = "0" * 64


@pytest.fixture(scope="module")
def loaders(prepared_config):
    return build_dataloaders(prepared_config)


def _confident_probabilities(y_true: np.ndarray) -> np.ndarray:
    """Well-separated probabilities, so thresholds are feasible everywhere."""
    probabilities = np.full((len(y_true), NUM_CLASSES), 0.02)
    probabilities[np.arange(len(y_true)), y_true] = 0.88
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _predictions_for(loaders, prepared_config, split: str) -> SplitPredictions:
    dataset = getattr(loaders, split).dataset
    y_true = np.array(dataset.labels)
    return build_split_predictions(
        loaders, split, prepared_config, y_true, _confident_probabilities(y_true)
    )


@pytest.fixture(scope="module")
def val_predictions(loaders, prepared_config):
    return _predictions_for(loaders, prepared_config, "val")


@pytest.fixture(scope="module")
def test_predictions(loaders, prepared_config):
    return _predictions_for(loaders, prepared_config, "test")


# --- construction ---------------------------------------------------------


def test_split_predictions_cannot_be_constructed_directly():
    # The tag is only trustworthy if it cannot be attached by hand.
    with pytest.raises(TypeError, match="must be built by build_split_predictions"):
        SplitPredictions(
            split="val",
            manifest_fingerprint="deadbeef",
            image_ids=("ISIC_0000001",),
            y_true=np.array([0]),
            y_prob=np.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES),
        )


def test_factory_tags_the_split_it_selected(val_predictions, test_predictions):
    assert val_predictions.split == "val"
    assert test_predictions.split == "test"
    assert val_predictions.manifest_fingerprint != test_predictions.manifest_fingerprint


def test_image_ids_belong_to_the_tagged_split(
    val_predictions, test_predictions, prepared_config
):
    # The drift guard: ids must come from the manifest the tag names.
    from ai.preprocessing.dataset import load_manifest

    for predictions in (val_predictions, test_predictions):
        manifest = load_manifest(prepared_config.manifest_path(predictions.split))
        assert set(predictions.image_ids) <= set(manifest["image_id"])
        assert len(predictions) == len(manifest)


def test_factory_rejects_predictions_from_another_split(
    loaders, prepared_config, test_predictions
):
    # Test arrays offered up as validation. Here the splits differ in size, so
    # the length guard fires first; the label guard below covers the case where
    # they do not.
    with pytest.raises(ValueError, match="must agree|did not come from"):
        build_split_predictions(
            loaders,
            "val",
            prepared_config,
            test_predictions.y_true,
            test_predictions.y_prob,
        )


def test_factory_rejects_labels_that_are_not_the_splits_own(
    loaders, prepared_config, val_predictions
):
    # Right length, wrong labels: only comparing against the split's manifest
    # labels catches this, and it is what proves the arrays came from `split`.
    foreign = np.asarray(val_predictions.y_true).copy()
    foreign[0] = (foreign[0] + 1) % NUM_CLASSES

    with pytest.raises(ValueError, match="did not come from 'val'"):
        build_split_predictions(
            loaders, "val", prepared_config, foreign, val_predictions.y_prob
        )


class _StubDataset:
    """Minimal stand-in exposing what the factory reads off a dataset."""

    def __init__(self, image_ids, labels):
        self.image_ids = list(image_ids)
        self.labels = list(labels)

    def __len__(self):
        return len(self.image_ids)


class _StubLoader:
    """A sequential loader over a stub dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.sampler = SequentialSampler(dataset)


def _loaders_yielding(split: str, image_ids, labels):
    """A `loaders`-shaped object whose split yields exactly these ids."""
    return SimpleNamespace(**{split: _StubLoader(_StubDataset(image_ids, labels))})


def _manifest_rows(prepared_config, split: str):
    from ai.preprocessing.dataset import load_manifest

    manifest = load_manifest(prepared_config.manifest_path(split))
    return list(manifest["image_id"]), [int(x) for x in manifest["label_idx"]]


def test_duplicate_loader_image_ids_rejected(prepared_config):
    ids, labels = _manifest_rows(prepared_config, "val")
    # Second row replaced by a copy of the first: still the right length, and
    # every id is in the manifest, so only a duplicate check catches it.
    duplicated_ids = [ids[0], ids[0], *ids[2:]]
    duplicated_labels = [labels[0], labels[0], *labels[2:]]
    loaders = _loaders_yielding("val", duplicated_ids, duplicated_labels)

    with pytest.raises(ValueError, match="duplicate loader ids"):
        build_split_predictions(
            loaders,
            "val",
            prepared_config,
            duplicated_labels,
            _confident_probabilities(np.array(duplicated_labels)),
        )


def test_loader_missing_a_manifest_image_rejected(prepared_config):
    ids, labels = _manifest_rows(prepared_config, "val")
    # A loader that silently drops an image: every id it yields is in the
    # manifest, so only the manifest-to-loader direction catches this.
    loaders = _loaders_yielding("val", ids[:-1], labels[:-1])

    with pytest.raises(ValueError, match="were not produced by the loader"):
        build_split_predictions(
            loaders,
            "val",
            prepared_config,
            labels[:-1],
            _confident_probabilities(np.array(labels[:-1])),
        )


def test_loader_with_an_unexpected_image_rejected(prepared_config):
    ids, labels = _manifest_rows(prepared_config, "val")
    extended_ids = [*ids, "ISIC_not_in_manifest"]
    extended_labels = [*labels, 0]
    loaders = _loaders_yielding("val", extended_ids, extended_labels)

    with pytest.raises(ValueError, match="absent from the manifest"):
        build_split_predictions(
            loaders,
            "val",
            prepared_config,
            extended_labels,
            _confident_probabilities(np.array(extended_labels)),
        )


def test_mismatch_error_names_the_split(prepared_config):
    ids, labels = _manifest_rows(prepared_config, "val")
    loaders = _loaders_yielding("val", ids[:-1], labels[:-1])

    with pytest.raises(ValueError, match="the 'val' loader and its manifest"):
        build_split_predictions(
            loaders,
            "val",
            prepared_config,
            labels[:-1],
            _confident_probabilities(np.array(labels[:-1])),
        )


def test_reordered_loader_ids_rejected(prepared_config):
    # Same images, same labels, different order. Set comparison passes this;
    # positional alignment would attach every prediction to the wrong row.
    ids, labels = _manifest_rows(prepared_config, "val")
    order = [1, 0, *range(2, len(ids))]
    shuffled_ids = [ids[i] for i in order]
    shuffled_labels = [labels[i] for i in order]
    loaders = _loaders_yielding("val", shuffled_ids, shuffled_labels)

    with pytest.raises(ValueError, match="different order"):
        build_split_predictions(
            loaders,
            "val",
            prepared_config,
            shuffled_labels,
            _confident_probabilities(np.array(shuffled_labels)),
        )


def test_matching_loader_still_accepted(loaders, prepared_config):
    # The tightened check must not reject the real, correct loader.
    predictions = _predictions_for(loaders, prepared_config, "val")

    ids, _ = _manifest_rows(prepared_config, "val")
    assert list(predictions.image_ids) == ids
    assert predictions.manifest_fingerprint == manifest_fingerprint(
        prepared_config.manifest_path("val")
    )


def test_factory_rejects_the_shuffled_training_loader(loaders, prepared_config):
    with pytest.raises(ValueError, match="split must be one of"):
        build_split_predictions(loaders, "train", prepared_config, [], [])


def test_prediction_arrays_are_read_only(val_predictions):
    with pytest.raises(ValueError):
        val_predictions.y_prob[0, 0] = 0.5


# --- fitting --------------------------------------------------------------


def test_fitting_on_test_is_refused(test_predictions):
    with pytest.raises(ValueError, match="may only be fitted on the 'val' split"):
        fit_thresholds(
            test_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
        )


def test_fitting_on_validation_is_allowed(val_predictions):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )

    assert thresholds.fitted_on == FITTING_SPLIT
    assert thresholds.manifest_fingerprint == val_predictions.manifest_fingerprint
    assert thresholds.thresholds.shape == (NUM_CLASSES,)
    assert thresholds.feasible.any()


def test_separable_scores_recover_a_usable_threshold():
    # Class 0 scores 0.9 when present and 0.1 otherwise, so any threshold in
    # between is perfect; the fitted one must sit inside that gap.
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.full((4, NUM_CLASSES), 0.1)
    y_prob[:2, 0] = 0.9
    y_prob[2:, 1] = 0.9

    thresholds = _fit_unchecked(y_true, y_prob, precision_floor=1.0)

    assert thresholds.feasible[0]
    assert 0.1 < thresholds.thresholds[0] <= 0.9
    assert thresholds.recall[0] == pytest.approx(1.0)
    assert thresholds.precision[0] == pytest.approx(1.0)


def test_absent_class_is_infeasible_not_zero():
    y_true = np.zeros(4, dtype=int)
    y_prob = np.full((4, NUM_CLASSES), 1.0 / NUM_CLASSES)

    thresholds = _fit_unchecked(y_true, y_prob, precision_floor=0.5)

    absent = [index for index in range(NUM_CLASSES) if index != 0]
    assert not thresholds.feasible[absent].any()
    assert np.isnan(thresholds.thresholds[absent]).all()
    assert not (thresholds.thresholds[absent] == 0.0).any()


def test_unreachable_precision_floor_is_infeasible():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.full((4, NUM_CLASSES), 1.0 / NUM_CLASSES)

    thresholds = _fit_unchecked(y_true, y_prob, precision_floor=1.0)

    assert not thresholds.feasible.any()


def test_precision_floor_is_validated(val_predictions):
    with pytest.raises(ValueError, match="precision_floor must be in"):
        fit_thresholds(
            val_predictions, precision_floor=1.5, checkpoint_fingerprint=CHECKPOINT
        )


def _synthetic(split: str, y_true, y_prob) -> SplitPredictions:
    """Hand-built predictions for cases the loader fixtures cannot produce.

    Bypasses the factory deliberately, to exercise the fitting arithmetic on
    arrays chosen for their edge-case shape rather than on the fixture split.
    """
    predictions = object.__new__(SplitPredictions)
    object.__setattr__(predictions, "split", split)
    object.__setattr__(predictions, "manifest_fingerprint", "synthetic")
    object.__setattr__(
        predictions, "image_ids", tuple(f"i{n}" for n in range(len(y_true)))
    )
    object.__setattr__(predictions, "y_true", np.asarray(y_true))
    object.__setattr__(predictions, "y_prob", np.asarray(y_prob, dtype=float))
    return predictions


def _fit_unchecked(y_true, y_prob, precision_floor):
    """Fit from hand-built arrays, bypassing the loader-backed factory."""
    return fit_thresholds(
        _synthetic(FITTING_SPLIT, y_true, y_prob),
        precision_floor=precision_floor,
        checkpoint_fingerprint=CHECKPOINT,
    )


# --- freezing and application --------------------------------------------


def test_threshold_set_is_immutable(val_predictions):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )

    with pytest.raises(AttributeError):
        thresholds.precision_floor = 0.9
    with pytest.raises(ValueError):
        thresholds.thresholds[0] = 0.1


def test_only_val_fitted_thresholds_may_be_applied(val_predictions, test_predictions):
    fitted = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )
    forged = ThresholdSet(
        fitted_on="test",
        manifest_fingerprint=fitted.manifest_fingerprint,
        checkpoint_fingerprint=fitted.checkpoint_fingerprint,
        precision_floor=fitted.precision_floor,
        thresholds=np.array(fitted.thresholds),
        precision=np.array(fitted.precision),
        recall=np.array(fitted.recall),
        feasible=np.array(fitted.feasible),
    )

    with pytest.raises(ValueError, match="only 'val'-fitted thresholds"):
        apply_thresholds(test_predictions, forged)


def test_applying_to_test_predicts_or_abstains(val_predictions, test_predictions):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )
    predicted = apply_thresholds(test_predictions, thresholds)

    assert predicted.shape == (len(test_predictions),)
    assert set(np.unique(predicted)) <= {-1, *range(NUM_CLASSES)}


def test_abstention_is_not_folded_into_a_prediction():
    # Nothing reaches any threshold, so every sample must abstain rather than
    # silently fall back to argmax.
    y_true = np.array([0, 1])
    y_prob = np.full((2, NUM_CLASSES), 1.0 / NUM_CLASSES)
    predictions = _synthetic("test", y_true, y_prob)
    thresholds = ThresholdSet(
        fitted_on=FITTING_SPLIT,
        manifest_fingerprint="synthetic",
        checkpoint_fingerprint=CHECKPOINT,
        precision_floor=0.5,
        thresholds=np.full(NUM_CLASSES, 0.99),
        precision=np.full(NUM_CLASSES, 1.0),
        recall=np.full(NUM_CLASSES, 1.0),
        feasible=np.ones(NUM_CLASSES, dtype=bool),
    )

    assert (apply_thresholds(predictions, thresholds) == -1).all()

    metrics = operating_point_metrics(predictions, thresholds)
    assert metrics["abstained"] == 2
    assert metrics["coverage"] == pytest.approx(0.0)
    # An abstained positive is a miss, not a free pass.
    assert metrics["per_class"][CLASS_CODES[0]]["recall"] == pytest.approx(0.0)


# --- serialisation --------------------------------------------------------


def test_threshold_set_round_trips_through_json(val_predictions, tmp_path):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )
    path = tmp_path / "thresholds.json"
    thresholds.save(path)

    restored = ThresholdSet.load(path)

    assert restored.fitted_on == thresholds.fitted_on
    assert restored.manifest_fingerprint == thresholds.manifest_fingerprint
    assert restored.precision_floor == thresholds.precision_floor
    np.testing.assert_array_equal(restored.feasible, thresholds.feasible)
    np.testing.assert_allclose(
        restored.thresholds, thresholds.thresholds, equal_nan=True
    )


def test_infeasible_threshold_serialises_as_null(tmp_path):
    y_true = np.zeros(4, dtype=int)
    y_prob = np.full((4, NUM_CLASSES), 1.0 / NUM_CLASSES)
    thresholds = _fit_unchecked(y_true, y_prob, precision_floor=0.5)

    path = tmp_path / "thresholds.json"
    thresholds.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    absent = payload["per_class"][CLASS_CODES[1]]
    assert absent["threshold"] is None
    assert absent["feasible"] is False
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_missing_threshold_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="--split val"):
        ThresholdSet.load(tmp_path / "absent.json")


def test_unreadable_threshold_file_is_rejected(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        ThresholdSet.load(path)


def test_threshold_file_from_another_label_mapping_is_rejected(
    val_predictions, tmp_path
):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )
    payload = thresholds.to_dict()
    del payload["per_class"][CLASS_CODES[0]]

    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different label mapping"):
        ThresholdSet.load(path)


def test_fitting_records_the_checkpoint_it_was_fitted_from(val_predictions):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )

    assert thresholds.checkpoint_fingerprint == CHECKPOINT


def test_checkpoint_fingerprint_is_required(val_predictions):
    with pytest.raises(ValueError, match="checkpoint_fingerprint is required"):
        fit_thresholds(
            val_predictions, precision_floor=0.5, checkpoint_fingerprint=""
        )


def test_checkpoint_provenance_survives_serialisation(val_predictions, tmp_path):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )
    path = tmp_path / "thresholds.json"
    thresholds.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_fingerprint"] == CHECKPOINT
    assert ThresholdSet.load(path).checkpoint_fingerprint == CHECKPOINT


def test_threshold_file_without_checkpoint_provenance_is_rejected(
    val_predictions, tmp_path
):
    # A file predating the provenance binding cannot be tied to any weights.
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )
    payload = thresholds.to_dict()
    del payload["checkpoint_fingerprint"]

    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing provenance"):
        ThresholdSet.load(path)


def test_operating_point_reports_checkpoint_provenance(
    val_predictions, test_predictions
):
    thresholds = fit_thresholds(
        val_predictions, precision_floor=0.5, checkpoint_fingerprint=CHECKPOINT
    )

    metrics = operating_point_metrics(test_predictions, thresholds)

    assert metrics["threshold_checkpoint_fingerprint"] == CHECKPOINT
    assert metrics["threshold_manifest_fingerprint"] == thresholds.manifest_fingerprint


def test_chunked_hashing_matches_the_reference_digest(tmp_path):
    """Chunked reads must produce exactly the one-shot SHA-256."""
    import hashlib

    from ai.training.thresholds import _HASH_CHUNK_BYTES

    for payload in (
        b"",
        b"short weights",
        # Spans several chunks, with a partial one at the end.
        bytes(range(256)) * ((_HASH_CHUNK_BYTES * 2 // 256) + 7),
    ):
        path = tmp_path / "checkpoint.pt"
        path.write_bytes(payload)
        assert checkpoint_fingerprint(path) == hashlib.sha256(payload).hexdigest()


def test_checkpoint_fingerprint_tracks_content(tmp_path):
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    first.write_bytes(b"weights-a")
    second.write_bytes(b"weights-a")
    assert checkpoint_fingerprint(first) == checkpoint_fingerprint(second)

    second.write_bytes(b"weights-b")
    assert checkpoint_fingerprint(first) != checkpoint_fingerprint(second)


def test_manifest_fingerprint_tracks_content(prepared_config, tmp_path):
    original = prepared_config.manifest_path("val")
    copy = tmp_path / "val.csv"
    copy.write_bytes(original.read_bytes())
    assert manifest_fingerprint(copy) == manifest_fingerprint(original)

    copy.write_text(original.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert manifest_fingerprint(copy) != manifest_fingerprint(original)
