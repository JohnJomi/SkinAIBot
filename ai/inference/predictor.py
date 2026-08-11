"""The inference engine: a validated checkpoint in, a structured result out.

Preprocessing is `build_eval_transforms` - the same object evaluation uses, not
a reimplementation of it. That is the single most important property of this
module: if inference resized or normalised even slightly differently from
training, every number it produced would be quietly wrong, and no unit test of
either half on its own would notice.

The architecture is not taken on trust from checkpoint metadata. `model_name`
decides which model timm builds, so an unrecognised value would have this
module construct an arbitrary network and load weights into it. Only the
EfficientNet-B4 family Phase 2.2 established is accepted.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ai.inference.images import describe, load_rgb
from ai.inference.provenance import verify_threshold_provenance
from ai.inference.result import (
    CheckpointProvenance,
    Prediction,
    ThresholdPrediction,
    TopKEntry,
    class_name,
)
from ai.preprocessing.config import DataConfig
from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.preprocessing.transforms import build_eval_transforms
from ai.training.checkpoints import load_checkpoint
from ai.training.config import ModelConfig
from ai.training.engine import resolve_device
from ai.training.metrics import class_probabilities, top_k_predictions
from ai.training.model import build_model
from ai.training.thresholds import (
    ThresholdSet,
    apply_thresholds_to_probabilities,
    checkpoint_fingerprint,
)

# The architectures Phase 2.2 established. Not a registry and not a plugin
# point: an allowlist of the one family this pipeline was trained and evaluated
# against, so a checkpoint cannot name something else and be believed.
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "tf_efficientnet_b4",
    "efficientnet_b4",
)

DEFAULT_TOP_K = 3


def validate_architecture(model_name: str) -> str:
    """Return the architecture of `model_name`, refusing unsupported ones."""
    if not model_name:
        raise ValueError(
            "checkpoint does not record a model_name, so its architecture "
            "cannot be verified"
        )

    architecture = model_name.partition(".")[0]
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"checkpoint names architecture {architecture!r}, which is not a "
            f"supported EfficientNet-B4 variant {list(SUPPORTED_ARCHITECTURES)}; "
            "refusing to build an arbitrary model from checkpoint metadata"
        )
    return architecture


class Predictor:
    """Runs a validated checkpoint over images.

    Construction does all the expensive and all the fallible work - loading,
    validating, building the transform - so `predict` is a hot path that either
    works or fails on the image alone.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_config: DataConfig,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.data_config = data_config
        self.device = resolve_device(device)

        # Refuses a checkpoint trained under a different class order.
        checkpoint = load_checkpoint(self.checkpoint_path)
        model_name = str(checkpoint.get("model_name") or "")
        architecture = validate_architecture(model_name)

        model = build_model(ModelConfig(name=model_name, pretrained=False))
        # The weights are already in memory; `load_checkpoint(path, model=...)`
        # would re-read the file, which is hundreds of megabytes for a B4.
        model.load_state_dict(checkpoint["model_state_dict"])

        self._model = model.to(self.device).eval()
        self._transform = build_eval_transforms(data_config)
        self._fingerprint = checkpoint_fingerprint(self.checkpoint_path)
        self._provenance = CheckpointProvenance(
            checkpoint=str(self.checkpoint_path),
            checkpoint_fingerprint=self._fingerprint,
            model_name=model_name,
            architecture=architecture,
            epoch=checkpoint.get("epoch"),
            stage_name=checkpoint.get("stage_name"),
            selection_metric=checkpoint.get("metric_name"),
            selection_metric_value=checkpoint.get("metric_value"),
            class_codes=tuple(checkpoint.get("class_codes", ())),
        )

    @property
    def checkpoint_fingerprint(self) -> str:
        """The SHA-256 of the loaded checkpoint."""
        return self._fingerprint

    @property
    def provenance(self) -> CheckpointProvenance:
        """Which weights this predictor is running."""
        return self._provenance

    def predict(
        self,
        source: str | Path | Image.Image,
        top_k: int = DEFAULT_TOP_K,
        thresholds: ThresholdSet | None = None,
        require_manifest: bool = False,
    ) -> Prediction:
        """Predict for one image.

        `thresholds` is opt-in and explicit. Nothing is ever discovered from
        the filesystem, and no operating point is ever fitted here.
        """
        return self.predict_batch(
            [source],
            top_k=top_k,
            thresholds=thresholds,
            require_manifest=require_manifest,
        )[0]

    def predict_batch(
        self,
        sources: list[str | Path | Image.Image],
        top_k: int = DEFAULT_TOP_K,
        thresholds: ThresholdSet | None = None,
        require_manifest: bool = False,
    ) -> list[Prediction]:
        """Predict for several images in one forward pass, in input order."""
        if not 1 <= top_k <= NUM_CLASSES:
            raise ValueError(f"top_k must be in [1, {NUM_CLASSES}], got {top_k}")
        if not sources:
            return []

        probabilities = self._probabilities(sources)

        verification = None
        threshold_choices: np.ndarray | None = None
        if thresholds is not None:
            verification = verify_threshold_provenance(
                thresholds,
                self._fingerprint,
                self.data_config,
                require_manifest=require_manifest,
            )
            # Shared with evaluation: one implementation, one set of semantics.
            threshold_choices = apply_thresholds_to_probabilities(
                probabilities, thresholds
            )

        ranked = top_k_predictions(probabilities, k=top_k)
        return [
            self._build_prediction(
                describe(source),
                probabilities[index],
                ranked[index],
                None if threshold_choices is None else int(threshold_choices[index]),
                verification,
            )
            for index, source in enumerate(sources)
        ]

    def _probabilities(self, sources: list[Any]) -> np.ndarray:
        """Model probabilities for each source, in order."""
        tensors = [self._transform(load_rgb(source)) for source in sources]
        batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode():
            logits = self._model(batch)
            probabilities = class_probabilities(logits)
        return probabilities.cpu().numpy()

    def _build_prediction(
        self,
        source: str,
        probabilities: np.ndarray,
        ranked: list[tuple[str, float]],
        threshold_choice: int | None,
        verification: Any,
    ) -> Prediction:
        predicted_index = int(probabilities.argmax())
        predicted_code = CLASS_CODES[predicted_index]

        threshold_prediction = None
        if threshold_choice is not None:
            abstained = threshold_choice < 0
            code = None if abstained else CLASS_CODES[threshold_choice]
            threshold_prediction = ThresholdPrediction(
                predicted_code=code,
                predicted_name=None if code is None else class_name(code),
                abstained=abstained,
                verification=verification,
            )

        return Prediction(
            source=source,
            # The argmax result is always reported, threshold or not.
            predicted_code=predicted_code,
            predicted_name=class_name(predicted_code),
            confidence=float(probabilities[predicted_index]),
            probabilities={
                code: float(probabilities[index])
                for index, code in enumerate(CLASS_CODES)
            },
            top_k=tuple(
                TopKEntry(code=code, name=class_name(code), probability=probability)
                for code, probability in ranked
            ),
            provenance=self._provenance,
            threshold_prediction=threshold_prediction,
        )
