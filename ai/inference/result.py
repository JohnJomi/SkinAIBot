"""The structured result of a single inference.

Frozen dataclasses rather than dictionaries: the field names are a contract
that Phase 3 will consume, and a typo in a dict key would surface as a missing
value rather than an error.

The argmax prediction and the threshold prediction are separate fields, never
merged. They answer different questions - "what is the model's best guess" and
"does anything clear the operating point" - and a caller that only wants the
first must not have it silently replaced by the second.
"""

from dataclasses import dataclass
from typing import Any

from ai.preprocessing.labels import CLASS_NAMES


@dataclass(frozen=True)
class TopKEntry:
    """One entry of the Top-K list."""

    code: str
    name: str
    probability: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"code": self.code, "name": self.name, "probability": self.probability}


@dataclass(frozen=True)
class CheckpointProvenance:
    """Which weights produced a prediction.

    Every field is read from the checkpoint itself; nothing is inferred, so a
    result never claims provenance the checkpoint does not actually carry.
    """

    checkpoint: str
    checkpoint_fingerprint: str
    model_name: str
    architecture: str
    epoch: int | None
    stage_name: str | None
    selection_metric: str | None
    selection_metric_value: float | None
    class_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "epoch": self.epoch,
            "stage_name": self.stage_name,
            "selection_metric": self.selection_metric,
            "selection_metric_value": self.selection_metric_value,
            "class_codes": list(self.class_codes),
        }


@dataclass(frozen=True)
class ThresholdVerification:
    """What was actually checked before an operating point was applied.

    `manifest_verified` is a fact, not an aspiration: when the validation
    manifest is not reachable it stays False and the expected fingerprint is
    still reported, so a reader can tell "matched" from "could not check".
    """

    fitted_on: str
    precision_floor: float
    checkpoint_fingerprint: str
    checkpoint_verified: bool
    expected_manifest_fingerprint: str
    current_manifest_fingerprint: str | None
    manifest_verified: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "fitted_on": self.fitted_on,
            "precision_floor": self.precision_floor,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_verified": self.checkpoint_verified,
            "expected_manifest_fingerprint": self.expected_manifest_fingerprint,
            "current_manifest_fingerprint": self.current_manifest_fingerprint,
            "manifest_verified": self.manifest_verified,
        }


@dataclass(frozen=True)
class ThresholdPrediction:
    """The operating-point decision, which may be an abstention."""

    predicted_code: str | None
    predicted_name: str | None
    abstained: bool
    verification: ThresholdVerification

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "predicted_code": self.predicted_code,
            "predicted_name": self.predicted_name,
            "abstained": self.abstained,
            "verification": self.verification.to_dict(),
        }


@dataclass(frozen=True)
class Prediction:
    """One image's prediction, with provenance."""

    source: str
    predicted_code: str
    predicted_name: str
    confidence: float
    probabilities: dict[str, float]
    top_k: tuple[TopKEntry, ...]
    provenance: CheckpointProvenance
    threshold_prediction: ThresholdPrediction | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, with the argmax fields always present."""
        return {
            "source": self.source,
            "predicted_code": self.predicted_code,
            "predicted_name": self.predicted_name,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "top_k": [entry.to_dict() for entry in self.top_k],
            "provenance": self.provenance.to_dict(),
            "threshold_prediction": (
                None
                if self.threshold_prediction is None
                else self.threshold_prediction.to_dict()
            ),
        }


def class_name(code: str) -> str:
    """The clinical name for a class code."""
    return CLASS_NAMES[code]
