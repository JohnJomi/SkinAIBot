"""Per-class operating points, fitted on validation and frozen thereafter.

Choosing a decision threshold is model selection: doing it on the test split
turns the held-out set into training signal and inflates every number derived
from it afterwards. Guarding that with `if split == "test": raise` is not
enough, because the failure in practice is not someone passing the string
"test" - it is a pair of arrays drifting away from the label describing them.

So the split tag travels welded to the data. `SplitPredictions` carries the
split it came from and the fingerprint of that split's manifest, and it can
only be built by `build_split_predictions`, which derives the tag from the same
variable that selects the loader and then checks the image ids against that
split's manifest. `fit_thresholds` accepts nothing else, and accepts it only
when the tag says validation.
"""

import hashlib
import json
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve
from torch.utils.data import SequentialSampler

from ai.preprocessing.config import DataConfig
from ai.preprocessing.dataset import load_manifest
from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES

# The only split thresholds may be fitted from.
FITTING_SPLIT = "val"

# Evaluating the training split through `build_dataloaders` would use the
# augmented transforms and a shuffled order, so it is neither aligned to the
# manifest nor repeatable. Evaluation covers the two deterministic splits.
EVALUABLE_SPLITS: tuple[str, ...] = ("val", "test")

# Guards `SplitPredictions.__init__`. Module-private on purpose: the factory is
# the supported way to build one.
_FACTORY_TOKEN = object()


def manifest_fingerprint(path: Path) -> str:
    """SHA-256 of a split manifest, identifying the exact split scored."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_only(array: np.ndarray) -> np.ndarray:
    """A private copy that cannot be mutated through the returned view."""
    copy = np.array(array, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True)
class SplitPredictions:
    """Model outputs for one split, carrying which split produced them.

    Build with `build_split_predictions`. Constructing this directly raises:
    the whole point of the type is that the tag cannot be attached by hand to
    arrays from somewhere else.
    """

    split: str
    manifest_fingerprint: str
    image_ids: tuple[str, ...]
    y_true: np.ndarray
    y_prob: np.ndarray
    token: InitVar[object] = None

    def __post_init__(self, token: object) -> None:
        """Reject construction that did not come through the factory."""
        if token is not _FACTORY_TOKEN:
            raise TypeError(
                "SplitPredictions must be built by build_split_predictions(); "
                "constructing it directly would let the split tag drift from "
                "the data it describes"
            )

    def __len__(self) -> int:
        return len(self.image_ids)


def build_split_predictions(
    loaders: Any,
    split: str,
    data_config: DataConfig,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> SplitPredictions:
    """The single supported way to build `SplitPredictions`.

    `split` selects the loader *and* becomes the tag, so the two cannot
    disagree. The image ids are then checked against that split's manifest,
    which catches arrays that came from somewhere else entirely.
    """
    if split not in EVALUABLE_SPLITS:
        raise ValueError(
            f"split must be one of {list(EVALUABLE_SPLITS)}, got {split!r}"
        )

    loader = getattr(loaders, split)

    # Alignment of image_ids to rows holds only for an unshuffled loader.
    if not isinstance(getattr(loader, "sampler", None), SequentialSampler):
        raise ValueError(
            f"the {split!r} loader is not sequential; per-image predictions "
            "would not line up with the manifest order"
        )

    image_ids = tuple(loader.dataset.image_ids)
    y_true = _read_only(np.asarray(y_true))
    y_prob = _read_only(np.asarray(y_prob, dtype=float))

    if not len(image_ids) == len(y_true) == len(y_prob):
        raise ValueError(
            f"{split!r}: {len(image_ids)} image ids, {len(y_true)} labels and "
            f"{len(y_prob)} probability rows must agree"
        )
    if y_prob.ndim != 2 or y_prob.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"expected probabilities of shape (n, {NUM_CLASSES}), "
            f"got {y_prob.shape}"
        )

    # The labels are the provenance evidence. `image_ids` come from the loader,
    # so they always match the tag by construction and prove nothing about the
    # arrays; `y_true` was produced by iterating that same loader, so if it
    # disagrees with the split's manifest labels, these predictions came from
    # somewhere else.
    expected_labels = np.asarray(getattr(loader.dataset, "labels"))
    if not np.array_equal(y_true, expected_labels):
        raise ValueError(
            f"the supplied labels do not match the {split!r} split; these "
            f"predictions did not come from {split!r}"
        )

    manifest_path = data_config.manifest_path(split)
    expected_ids = set(load_manifest(manifest_path)["image_id"])
    unexpected = sorted(set(image_ids) - expected_ids)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} images in the {split!r} loader are absent from "
            f"its manifest; first: {unexpected[:5]}. The loader and the "
            "configured manifest disagree"
        )

    return SplitPredictions(
        split=split,
        manifest_fingerprint=manifest_fingerprint(manifest_path),
        image_ids=image_ids,
        y_true=y_true,
        y_prob=y_prob,
        token=_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class ThresholdSet:
    """Per-class decision thresholds and the run that produced them.

    Frozen, with read-only arrays: a threshold set applied to the test split
    must be exactly the one validation produced.
    """

    fitted_on: str
    manifest_fingerprint: str
    precision_floor: float
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    feasible: np.ndarray

    def __post_init__(self) -> None:
        """Freeze the arrays so a caller cannot edit an applied operating point."""
        for name in ("thresholds", "precision", "recall", "feasible"):
            object.__setattr__(self, name, _read_only(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; undefined thresholds become null, never 0.0."""
        return {
            "fitted_on": self.fitted_on,
            "manifest_fingerprint": self.manifest_fingerprint,
            "precision_floor": self.precision_floor,
            "per_class": {
                code: {
                    "threshold": _nullable(self.thresholds[index]),
                    "precision": _nullable(self.precision[index]),
                    "recall": _nullable(self.recall[index]),
                    "feasible": bool(self.feasible[index]),
                }
                for index, code in enumerate(CLASS_CODES)
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThresholdSet":
        """Rebuild from `to_dict` output; null becomes NaN again."""
        per_class = payload["per_class"]
        missing = [code for code in CLASS_CODES if code not in per_class]
        if missing:
            raise ValueError(
                f"threshold set is missing classes {missing}; it was fitted "
                "under a different label mapping"
            )
        return cls(
            fitted_on=payload["fitted_on"],
            manifest_fingerprint=payload["manifest_fingerprint"],
            precision_floor=float(payload["precision_floor"]),
            thresholds=np.array(
                [_from_nullable(per_class[c]["threshold"]) for c in CLASS_CODES]
            ),
            precision=np.array(
                [_from_nullable(per_class[c]["precision"]) for c in CLASS_CODES]
            ),
            recall=np.array(
                [_from_nullable(per_class[c]["recall"]) for c in CLASS_CODES]
            ),
            feasible=np.array(
                [bool(per_class[c]["feasible"]) for c in CLASS_CODES], dtype=bool
            ),
        )

    def save(self, path: Path) -> None:
        """Write the threshold set as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "ThresholdSet":
        """Read a threshold set. A missing or unreadable file is an error."""
        if not path.is_file():
            raise FileNotFoundError(
                f"threshold file {path} not found; produce one with "
                "`python -m ai.training.evaluate --split val`"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"threshold file {path} is not valid JSON: {error}"
            ) from None
        return cls.from_dict(payload)


def _nullable(value: float) -> float | None:
    return None if value != value else float(value)


def _from_nullable(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def fit_thresholds(
    predictions: SplitPredictions, *, precision_floor: float
) -> ThresholdSet:
    """Fit per-class thresholds maximising recall subject to a precision floor.

    Only validation predictions are accepted. There is no split argument to
    misstate: the tag arrives attached to the data.

    Recall is what the medical setting cares about, so among the thresholds
    meeting the precision floor we take the one with the highest recall. A
    class with no positives, or none meeting the floor, is marked infeasible
    with a NaN threshold rather than being given a misleading number.
    """
    if predictions.split != FITTING_SPLIT:
        raise ValueError(
            f"thresholds may only be fitted on the {FITTING_SPLIT!r} split; "
            f"these predictions are from {predictions.split!r}. Fitting on "
            "test would turn the held-out split into model selection"
        )
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError(
            f"precision_floor must be in [0, 1], got {precision_floor}"
        )

    y_true = np.asarray(predictions.y_true)
    y_prob = np.asarray(predictions.y_prob)

    thresholds = np.full(NUM_CLASSES, np.nan)
    precisions = np.full(NUM_CLASSES, np.nan)
    recalls = np.full(NUM_CLASSES, np.nan)
    feasible = np.zeros(NUM_CLASSES, dtype=bool)

    for class_index in range(NUM_CLASSES):
        positives = y_true == class_index
        n_positive = int(positives.sum())
        if n_positive == 0 or n_positive == len(y_true):
            continue

        precision, recall, curve = precision_recall_curve(
            positives.astype(int), y_prob[:, class_index]
        )
        # precision_recall_curve appends a (1, 0) point with no threshold.
        precision, recall = precision[:-1], recall[:-1]

        meets_floor = precision >= precision_floor
        if not meets_floor.any():
            continue

        candidates = np.flatnonzero(meets_floor)
        best = candidates[int(np.argmax(recall[candidates]))]
        thresholds[class_index] = float(curve[best])
        precisions[class_index] = float(precision[best])
        recalls[class_index] = float(recall[best])
        feasible[class_index] = True

    return ThresholdSet(
        fitted_on=predictions.split,
        manifest_fingerprint=predictions.manifest_fingerprint,
        precision_floor=float(precision_floor),
        thresholds=thresholds,
        precision=precisions,
        recall=recalls,
        feasible=feasible,
    )


def apply_thresholds(
    predictions: SplitPredictions, thresholds: ThresholdSet
) -> np.ndarray:
    """Predicted class per sample under the operating point, -1 when none fires.

    A sample where no feasible class reaches its threshold is left undecided
    rather than being forced into its argmax class: an explicit abstention is
    the useful signal in triage, and folding it into a prediction would hide it.
    """
    if thresholds.fitted_on != FITTING_SPLIT:
        raise ValueError(
            f"threshold set was fitted on {thresholds.fitted_on!r}; only "
            f"{FITTING_SPLIT!r}-fitted thresholds may be applied"
        )

    y_prob = np.asarray(predictions.y_prob)
    limits = np.where(thresholds.feasible, thresholds.thresholds, np.inf)

    qualifies = y_prob >= limits
    # Rank by how far past its threshold each class is, so classes with
    # different thresholds compare fairly.
    margin = np.where(qualifies, y_prob - limits, -np.inf)

    predicted = np.full(len(y_prob), -1, dtype=int)
    any_qualifies = qualifies.any(axis=1)
    predicted[any_qualifies] = margin[any_qualifies].argmax(axis=1)
    return predicted


def operating_point_metrics(
    predictions: SplitPredictions, thresholds: ThresholdSet
) -> dict[str, Any]:
    """Per-class precision/recall under the operating point, plus coverage.

    Computed directly rather than through scikit-learn because abstentions
    (-1) are not a class: an abstained positive is a false negative, and no
    sample abstained on is ever a false positive.
    """
    predicted = apply_thresholds(predictions, thresholds)
    y_true = np.asarray(predictions.y_true)
    total = len(y_true)

    per_class: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    for class_index, code in enumerate(CLASS_CODES):
        actual = y_true == class_index
        called = predicted == class_index
        true_positive = int((actual & called).sum())
        false_positive = int((~actual & called).sum())
        false_negative = int((actual & ~called).sum())

        recall = (
            true_positive / (true_positive + false_negative)
            if actual.any()
            else float("nan")
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if called.any()
            else float("nan")
        )
        if actual.any():
            recalls.append(recall)

        per_class[code] = {
            "threshold": _nullable(thresholds.thresholds[class_index]),
            "feasible": bool(thresholds.feasible[class_index]),
            "recall": recall,
            "precision": precision,
            "support": int(actual.sum()),
            "predicted": int(called.sum()),
        }

    abstained = int((predicted == -1).sum())
    return {
        "fitted_on": thresholds.fitted_on,
        "precision_floor": thresholds.precision_floor,
        "threshold_manifest_fingerprint": thresholds.manifest_fingerprint,
        "coverage": (total - abstained) / total if total else float("nan"),
        "abstained": abstained,
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "per_class": per_class,
    }
