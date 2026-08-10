"""One pass over a DataLoader, for training and for evaluation.

Kept free of CLIs, file I/O and configuration so the loops can be unit tested
directly. Neither function knows which split it was handed - the caller does.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ai.training.config import StageConfig
from ai.training.metrics import class_probabilities
from ai.training.model import set_train_mode


def resolve_device(name: str) -> torch.device:
    """Turn the configured device name into a real device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: StageConfig,
) -> float:
    """Train for one epoch; return the mean loss per sample.

    `set_train_mode` rather than `model.train()`: the frozen backbone's
    BatchNorm layers must stay in eval mode for this stage.
    """
    set_train_mode(model, stage)

    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach()) * targets.size(0)
        total_samples += targets.size(0)

    if total_samples == 0:
        raise ValueError("training loader yielded no samples")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the model over `loader`; return (mean loss, y_true, y_prob).

    Returns raw arrays rather than metrics so the caller decides what to
    compute - validation wants one number, the final report wants everything.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    targets_seen: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += float(loss.detach()) * targets.size(0)
        total_samples += targets.size(0)
        targets_seen.append(targets.cpu().numpy())
        probabilities.append(class_probabilities(logits).cpu().numpy())

    if total_samples == 0:
        raise ValueError("evaluation loader yielded no samples")

    return (
        total_loss / total_samples,
        np.concatenate(targets_seen),
        np.concatenate(probabilities),
    )
