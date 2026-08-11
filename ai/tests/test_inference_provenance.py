"""Threshold provenance is verified, and never claimed when it was not."""

import numpy as np
import pytest

from ai.inference.provenance import verify_threshold_provenance
from ai.preprocessing.labels import NUM_CLASSES
from ai.training.thresholds import ThresholdSet, manifest_fingerprint

CHECKPOINT = "a" * 64


def _thresholds(manifest_fingerprint_value, fitted_on="val", checkpoint=CHECKPOINT):
    return ThresholdSet(
        fitted_on=fitted_on,
        manifest_fingerprint=manifest_fingerprint_value,
        checkpoint_fingerprint=checkpoint,
        precision_floor=0.5,
        thresholds=np.full(NUM_CLASSES, 0.5),
        precision=np.full(NUM_CLASSES, 1.0),
        recall=np.full(NUM_CLASSES, 1.0),
        feasible=np.ones(NUM_CLASSES, dtype=bool),
    )


@pytest.fixture
def val_fingerprint(prepared_config):
    return manifest_fingerprint(prepared_config.manifest_path("val"))


@pytest.fixture
def config_without_manifests(prepared_config, tmp_path):
    """A deployment-shaped config: no data manifests on disk."""
    return prepared_config.model_copy(update={"manifest_dir": tmp_path / "absent"})


def test_matching_manifest_is_verified(prepared_config, val_fingerprint):
    verification = verify_threshold_provenance(
        _thresholds(val_fingerprint), CHECKPOINT, prepared_config
    )

    assert verification.manifest_verified is True
    assert verification.checkpoint_verified is True
    assert verification.current_manifest_fingerprint == val_fingerprint
    assert verification.expected_manifest_fingerprint == val_fingerprint
    assert verification.fitted_on == "val"


def test_stale_manifest_fingerprint_is_rejected(prepared_config):
    with pytest.raises(ValueError, match="different validation split"):
        verify_threshold_provenance(
            _thresholds("0" * 64), CHECKPOINT, prepared_config
        )


def test_wrong_checkpoint_is_rejected(prepared_config, val_fingerprint):
    with pytest.raises(ValueError, match="different checkpoint"):
        verify_threshold_provenance(
            _thresholds(val_fingerprint), "f" * 64, prepared_config
        )


def test_thresholds_not_fitted_on_val_are_rejected(prepared_config, val_fingerprint):
    with pytest.raises(ValueError, match="only 'val'-fitted"):
        verify_threshold_provenance(
            _thresholds(val_fingerprint, fitted_on="test"), CHECKPOINT, prepared_config
        )


def test_unavailable_manifest_is_recorded_not_assumed(
    config_without_manifests, val_fingerprint
):
    verification = verify_threshold_provenance(
        _thresholds(val_fingerprint), CHECKPOINT, config_without_manifests
    )

    # The checks that are possible still ran; the one that is not is reported
    # as not run, never as passed.
    assert verification.checkpoint_verified is True
    assert verification.manifest_verified is False
    assert verification.current_manifest_fingerprint is None
    # The expected value is still published, so the gap is auditable.
    assert verification.expected_manifest_fingerprint == val_fingerprint


def test_unavailable_manifest_is_fatal_when_required(
    config_without_manifests, val_fingerprint
):
    with pytest.raises(ValueError, match="is not available"):
        verify_threshold_provenance(
            _thresholds(val_fingerprint),
            CHECKPOINT,
            config_without_manifests,
            require_manifest=True,
        )


def test_checkpoint_mismatch_beats_a_missing_manifest(config_without_manifests):
    # The checkpoint check must not be skipped just because the manifest is
    # unavailable.
    with pytest.raises(ValueError, match="different checkpoint"):
        verify_threshold_provenance(
            _thresholds("0" * 64, checkpoint="b" * 64),
            CHECKPOINT,
            config_without_manifests,
        )


def test_verification_serialises_faithfully(prepared_config, val_fingerprint):
    payload = verify_threshold_provenance(
        _thresholds(val_fingerprint), CHECKPOINT, prepared_config
    ).to_dict()

    assert payload["manifest_verified"] is True
    assert payload["precision_floor"] == 0.5
    assert payload["checkpoint_fingerprint"] == CHECKPOINT
