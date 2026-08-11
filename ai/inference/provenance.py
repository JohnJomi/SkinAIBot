"""Threshold provenance verification at inference time.

Phase 2.3 binds a `ThresholdSet` to three things: the split it was fitted on,
the checkpoint it was fitted for, and the validation manifest that chose it.
Two of those can always be checked here. The third cannot: a deployment does
not ship `ai/data_manifests/val.csv`, so requiring it unconditionally would
make thresholds unusable outside a training checkout.

The resolution is to check it when it is reachable and to *say so* when it is
not. `manifest_verified` is never set to True on the strength of an assumption,
and `--require-manifest-check` turns "not reachable" into a hard failure for
callers that need the stronger guarantee.
"""

from ai.inference.result import ThresholdVerification
from ai.preprocessing.config import DataConfig
from ai.training.thresholds import (
    FITTING_SPLIT,
    ThresholdSet,
    manifest_fingerprint,
)


def verify_threshold_provenance(
    thresholds: ThresholdSet,
    checkpoint_fingerprint: str,
    data_config: DataConfig,
    require_manifest: bool = False,
) -> ThresholdVerification:
    """Check a threshold set against the run that is about to use it.

    Raises `ValueError` on any mismatch. Returns a record of what was verified
    so the caller can put it in the result rather than restating it.
    """
    if thresholds.fitted_on != FITTING_SPLIT:
        raise ValueError(
            f"threshold set was fitted on {thresholds.fitted_on!r}; only "
            f"{FITTING_SPLIT!r}-fitted thresholds may be applied"
        )

    if thresholds.checkpoint_fingerprint != checkpoint_fingerprint:
        raise ValueError(
            "threshold set was fitted from a different checkpoint "
            f"(fingerprint {thresholds.checkpoint_fingerprint[:12]}..., current "
            f"{checkpoint_fingerprint[:12]}...); an operating point belongs to "
            "the weights it was fitted for"
        )

    manifest_path = data_config.manifest_path(FITTING_SPLIT)
    if manifest_path.is_file():
        current = manifest_fingerprint(manifest_path)
        if thresholds.manifest_fingerprint != current:
            raise ValueError(
                "threshold set was fitted against a different validation split "
                f"(fingerprint {thresholds.manifest_fingerprint[:12]}..., "
                f"current {current[:12]}...); refit with "
                f"`--split {FITTING_SPLIT}`"
            )
        manifest_verified = True
    else:
        if require_manifest:
            raise ValueError(
                f"validation manifest {manifest_path} is not available, so the "
                "threshold set's validation fingerprint cannot be verified; "
                "drop --require-manifest-check to proceed on the checkpoint "
                "and split checks alone"
            )
        current = None
        manifest_verified = False

    return ThresholdVerification(
        fitted_on=thresholds.fitted_on,
        precision_floor=thresholds.precision_floor,
        checkpoint_fingerprint=checkpoint_fingerprint,
        checkpoint_verified=True,
        expected_manifest_fingerprint=thresholds.manifest_fingerprint,
        current_manifest_fingerprint=current,
        manifest_verified=manifest_verified,
    )
