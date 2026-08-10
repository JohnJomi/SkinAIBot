"""Class imbalance utilities.

HAM10000 is severely imbalanced - `nv` accounts for roughly two thirds of the
images while `df` accounts for around one percent, a ratio near 58:1. A model
trained on it unaddressed can reach high accuracy by predicting `nv`.

Two standard remedies are offered here, and they are deliberately *not*
combined: resampling and loss weighting both correct the same imbalance, so
applying them together over-corrects the rare classes. Phase 2.2 picks one.
Neither is enabled by default.
"""

import torch
from torch.utils.data import WeightedRandomSampler

from ai.preprocessing.labels import NUM_CLASSES


def class_counts(labels: list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Number of samples per class index, as a length-`num_classes` tensor."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for label in labels:
        counts[label] += 1
    return counts


def compute_class_weights(
    labels: list[int], num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """Inverse-frequency weights for a weighted loss function.

    Normalised to mean 1.0 so that swapping weighting on or off does not also
    change the effective learning rate. Absent classes receive weight 0.
    """
    counts = class_counts(labels, num_classes)
    present = counts > 0
    if not present.any():
        raise ValueError("cannot compute class weights from an empty label set")

    weights = torch.zeros(num_classes, dtype=torch.float)
    weights[present] = counts[present].sum().float() / counts[present].float()
    weights[present] = weights[present] / weights[present].mean()
    return weights


def build_weighted_sampler(
    labels: list[int],
    generator: torch.Generator | None = None,
    num_classes: int = NUM_CLASSES,
) -> WeightedRandomSampler:
    """Sampler that draws classes with equal probability, with replacement.

    One epoch keeps its original length; rare classes are simply revisited more
    often within it.
    """
    counts = class_counts(labels, num_classes)
    if len(labels) == 0:
        raise ValueError("cannot build a sampler from an empty label set")

    per_sample = [1.0 / float(counts[label]) for label in labels]
    return WeightedRandomSampler(
        weights=per_sample,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
