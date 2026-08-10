"""Checkpoints round-trip, carry their label order, and track the best epoch."""

import pytest
import torch
from torch import nn

from ai.preprocessing.labels import CLASS_CODES, NUM_CLASSES
from ai.training.checkpoints import (
    BestCheckpointTracker,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def saved(tmp_path):
    """A checkpoint written from a tiny model, plus that model."""
    model = nn.Linear(4, NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One step, so the optimizer has real state to serialise.
    model(torch.zeros(2, 4)).sum().backward()
    optimizer.step()

    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        stage_name="finetune",
        metric_name="macro_recall",
        metric_value=0.42,
        model_name="tf_efficientnet_b4.aa_in1k",
        config_snapshot={"monitor": "macro_recall"},
    )
    return path, model


def test_checkpoint_records_the_run_it_came_from(saved):
    path, _ = saved
    checkpoint = load_checkpoint(path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["stage_name"] == "finetune"
    assert checkpoint["metric_name"] == "macro_recall"
    assert checkpoint["metric_value"] == pytest.approx(0.42)
    assert checkpoint["model_name"] == "tf_efficientnet_b4.aa_in1k"
    assert checkpoint["class_codes"] == list(CLASS_CODES)


def test_loading_restores_the_saved_weights(saved):
    path, model = saved
    original = model.weight.detach().clone()

    restored = nn.Linear(4, NUM_CLASSES)
    with torch.no_grad():
        restored.weight.zero_()
    assert not torch.equal(restored.weight, original)

    load_checkpoint(path, model=restored)
    assert torch.equal(restored.weight, original)


def test_checkpoint_from_a_different_label_order_is_refused(saved, monkeypatch):
    path, _ = saved
    # Same tensor shapes, different meaning: nothing else would catch this.
    monkeypatch.setattr(
        "ai.training.checkpoints.CLASS_CODES", ("vasc", *CLASS_CODES[1:])
    )

    with pytest.raises(ValueError, match="silently relabel every prediction"):
        load_checkpoint(path)


def test_missing_checkpoint_reports_how_to_produce_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="ai.training.train"):
        load_checkpoint(tmp_path / "absent.pt")


def test_tracker_accepts_only_improvements():
    tracker = BestCheckpointTracker(metric_name="macro_recall")

    assert tracker.update(0.10, epoch=1) is True
    assert tracker.update(0.05, epoch=2) is False
    assert tracker.update(0.10, epoch=3) is False
    assert tracker.update(0.20, epoch=4) is True

    assert tracker.best_value == pytest.approx(0.20)
    assert tracker.best_epoch == 4


def test_tracker_never_treats_nan_as_an_improvement():
    tracker = BestCheckpointTracker(metric_name="macro_auc")

    assert tracker.update(float("nan"), epoch=1) is False
    assert tracker.best_epoch is None

    assert tracker.update(0.5, epoch=2) is True
    assert tracker.update(float("nan"), epoch=3) is False
    assert tracker.best_epoch == 2
