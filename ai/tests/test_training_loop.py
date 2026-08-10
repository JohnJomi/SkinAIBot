"""Deterministic smoke test for the staged training loop.

Nothing here asserts that the loss goes down: on ~90 synthetic images with
random labels that is a coin flip, and a flaky guard is worse than none. The
assertions are all about control flow and side effects - what ran, what
changed, what was written, and what was never touched.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from ai.preprocessing.datamodule import DataLoaders, build_dataloaders
from ai.preprocessing.labels import NUM_CLASSES
from ai.training import train as train_module
from ai.training.checkpoints import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    load_checkpoint,
)
from ai.training.config import ModelConfig, StageConfig, TrainingConfig
from ai.training.train import (
    HISTORY_NAME,
    configure_determinism,
    run_training,
    seed_everything,
)


class TinyNet(nn.Module):
    """A timm-shaped stand-in: `blocks`, `conv_head`/`bn2`, `get_classifier`.

    Small enough that two epochs finish in well under a second, while still
    exercising the real freezing and BatchNorm handling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)),
                nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4)),
            ]
        )
        self.conv_head = nn.Conv2d(4, 8, 1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, NUM_CLASSES)
        self.forward_calls = 0

    def get_classifier(self) -> nn.Module:
        """The head, as timm exposes it."""
        return self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stub, counting calls so the smoke test can assert on them."""
        self.forward_calls += 1
        for block in self.blocks:
            x = block(x)
        x = self.bn2(self.conv_head(x))
        return self.classifier(self.pool(x).flatten(1))


class LoaderTripwire:
    """Stands in for `loaders.test` and fails loudly on any access.

    Covers the loader *and* the underlying dataset, so a pixel read is caught
    even if it bypassed iteration.
    """

    def __init__(self) -> None:
        self.accesses: list[str] = []

    def _trip(self, how: str):
        """Record the access and fail immediately."""
        self.accesses.append(how)
        raise AssertionError(f"train.py accessed the test split via {how}")

    def __iter__(self):
        self._trip("__iter__")

    def __len__(self):
        self._trip("__len__")

    def __getitem__(self, index):
        self._trip("__getitem__")

    @property
    def dataset(self):
        self._trip("dataset")


class CountingLoader:
    """Delegates to a real DataLoader, counting how often it is iterated."""

    def __init__(self, loader) -> None:
        self._loader = loader
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset


def _training_config(tmp_path: Path) -> TrainingConfig:
    """Two stages, one epoch each: enough to cover a stage transition."""
    return TrainingConfig(
        model=ModelConfig(name="stub", pretrained=False),
        stages=(
            StageConfig(
                name="head", epochs=1, lr=1e-2, weight_decay=0.0, unfreeze_blocks=0
            ),
            StageConfig(
                name="finetune",
                epochs=1,
                lr=1e-3,
                weight_decay=0.0,
                unfreeze_blocks=1,
                scheduler="none",
            ),
        ),
        imbalance_strategy="weighted_loss",
        monitor="macro_recall",
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        device="cpu",
    )


@pytest.fixture
def trained(prepared_config, tmp_path, monkeypatch):
    """Run `run_training` against the synthetic dataset with a stub model."""
    model = TinyNet()
    captured: dict = {"model": model}

    monkeypatch.setattr("ai.training.train.build_model", lambda config: model)

    real_build = DataLoaders  # keep the dataclass; only swap the test loader
    from ai.preprocessing.datamodule import build_dataloaders as real_builder

    def instrumented(config, use_weighted_sampler=False):
        """Real loaders, with the test split swapped for a tripwire."""
        loaders = real_builder(config, use_weighted_sampler=use_weighted_sampler)
        captured["tripwire"] = LoaderTripwire()
        captured["val"] = CountingLoader(loaders.val)
        captured["train_batches"] = len(loaders.train)
        return real_build(
            train=loaders.train,
            val=captured["val"],
            test=captured["tripwire"],
            train_labels=loaders.train_labels,
        )

    monkeypatch.setattr("ai.training.train.build_dataloaders", instrumented)

    steps: list[int] = []
    original_step = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        """Count optimizer steps, then delegate."""
        steps.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    # Counting via Tensor.backward rather than a module backward hook: the hook
    # warns whenever no input requires grad, which is exactly the frozen-stem
    # case Stage 1 creates.
    backward_calls: list[int] = []
    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        """Count backward passes, then delegate."""
        backward_calls.append(1)
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)

    config = _training_config(tmp_path)
    frozen_before = model.blocks[0][0].weight.detach().clone()
    head_before = model.classifier.weight.detach().clone()

    history = run_training(prepared_config, config)

    captured.update(
        history=history,
        config=config,
        steps=steps,
        backward_calls=backward_calls,
        frozen_before=frozen_before,
        head_before=head_before,
    )
    return captured


def test_training_completes_the_configured_epochs(trained):
    history = trained["history"]
    config = trained["config"]

    assert len(history) == config.total_epochs == 2
    assert [entry["epoch"] for entry in history] == [1, 2]
    assert [entry["stage"] for entry in history] == ["head", "finetune"]


def test_forward_backward_and_optimizer_steps_execute(trained):
    expected_train_steps = trained["train_batches"] * trained["config"].total_epochs

    assert len(trained["steps"]) == expected_train_steps
    assert len(trained["backward_calls"]) == expected_train_steps
    # Training batches plus the validation passes.
    assert trained["model"].forward_calls > expected_train_steps


def test_trainable_parameters_update_and_frozen_ones_do_not(trained):
    model = trained["model"]

    assert not torch.equal(model.classifier.weight, trained["head_before"])
    # blocks[0] is outside unfreeze_blocks in both stages, so it must be
    # bit-identical - proof that freezing held for the whole run.
    assert torch.equal(model.blocks[0][0].weight, trained["frozen_before"])


def test_validation_runs_every_epoch(trained):
    history = trained["history"]

    assert trained["val"].iterations == trained["config"].total_epochs
    for entry in history:
        assert "val_loss" in entry
        assert entry["monitored"] == entry["val_macro_recall"]


def test_best_and_last_checkpoints_are_written(trained):
    checkpoint_dir = trained["config"].checkpoint_dir

    last = load_checkpoint(checkpoint_dir / LAST_CHECKPOINT_NAME)
    best = load_checkpoint(checkpoint_dir / BEST_CHECKPOINT_NAME)

    assert last["epoch"] == trained["config"].total_epochs
    assert best["metric_name"] == trained["config"].monitor
    assert best["epoch"] <= last["epoch"]
    assert (checkpoint_dir / HISTORY_NAME).is_file()


def test_train_never_touches_the_test_split(trained):
    # The guarantee the split exists to provide: training saw no test image.
    assert trained["tripwire"].accesses == []


def test_undefined_macro_auc_is_written_as_null(prepared_config, tmp_path, monkeypatch):
    # Every class is present in the synthetic validation split, so macro_auc is
    # always defined there; force the undefined case to cover the encoding.
    model = TinyNet()
    monkeypatch.setattr("ai.training.train.build_model", lambda config: model)

    real_compute = train_module.compute_metrics

    def undefined_auc(y_true, y_prob):
        """Real metrics, but with macro_auc forced undefined."""
        metrics = real_compute(y_true, y_prob)
        metrics["macro_auc"] = float("nan")
        return metrics

    monkeypatch.setattr("ai.training.train.compute_metrics", undefined_auc)

    config = _training_config(tmp_path)
    run_training(prepared_config, config)

    text = (config.checkpoint_dir / HISTORY_NAME).read_text(encoding="utf-8")
    # A bare NaN literal is not valid JSON, whatever Python's parser tolerates.
    assert "NaN" not in text
    assert all(entry["val_macro_auc"] is None for entry in json.loads(text))


def test_seeding_reproduces_initial_weights(prepared_config):
    seed_everything(prepared_config.seed)
    first = TinyNet().classifier.weight.detach().clone()

    seed_everything(prepared_config.seed)
    second = TinyNet().classifier.weight.detach().clone()

    assert torch.equal(first, second)


def test_seeding_reproduces_dataloader_order(prepared_config):
    """The shuffled training order must depend only on the seed."""

    def first_batch_targets():
        seed_everything(prepared_config.seed)
        loaders = build_dataloaders(prepared_config)
        return next(iter(loaders.train))[1]

    assert torch.equal(first_batch_targets(), first_batch_targets())


@pytest.fixture
def determinism_state():
    """Save and restore the global determinism flags around a test."""
    saved = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    yield
    torch.backends.cudnn.deterministic = saved[0]
    torch.backends.cudnn.benchmark = saved[1]
    torch.use_deterministic_algorithms(saved[2], warn_only=True)
    if saved[3] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved[3]


def test_cuda_determinism_flags_are_set(determinism_state):
    # No GPU needed: these are global settings, and nothing is allocated.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    configure_determinism(torch.device("cuda"))

    assert torch.backends.cudnn.deterministic is True
    # Benchmarking picks a convolution algorithm per shape; that choice is the
    # nondeterminism, so it has to be off.
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_operator_supplied_cublas_workspace_is_kept(determinism_state):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    configure_determinism(torch.device("cuda"))

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cpu_and_mps_behaviour_is_untouched(device, determinism_state):
    # use_deterministic_algorithms is global and turns CPU ops that lack a
    # deterministic kernel into hard errors, so these devices are left alone.
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    configure_determinism(torch.device(device))

    assert torch.backends.cudnn.benchmark is True
    assert torch.are_deterministic_algorithms_enabled() is False
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
