"""EfficientNet-B4 construction, head replacement and stage freezing.

The number of output classes is derived from the Phase 2.1 label mapping, so
the model and the manifests can never disagree about how wide the output is.

Freezing is two separate concerns, and conflating them is the classic bug:

- `requires_grad=False` stops a parameter being *updated*;
- it does NOT stop a BatchNorm layer updating `running_mean`/`running_var`,
  which happens in the forward pass regardless.

So a "frozen" backbone whose BatchNorms are left in train mode keeps drifting
its normalisation statistics while the classification head is still random.
`set_train_mode` therefore re-asserts eval mode on every frozen BatchNorm after
each `model.train()` call, not once at the start of a stage.
"""

from collections.abc import Iterator

import timm
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ai.preprocessing.labels import NUM_CLASSES
from ai.training.config import ModelConfig, StageConfig

# EfficientNet's post-block projection, unfrozen alongside the final blocks.
FINAL_MODULE_NAMES: tuple[str, ...] = ("conv_head", "bn2")


def build_model(config: ModelConfig) -> nn.Module:
    """Build the backbone with a fresh head sized to the canonical classes.

    Passing `num_classes` to timm *is* the head replacement: it constructs a
    new, randomly initialised classifier in place of the 1000-way ImageNet one.
    """
    model = timm.create_model(
        config.name,
        pretrained=config.pretrained,
        num_classes=NUM_CLASSES,
        drop_rate=config.drop_rate,
    )

    classifier = model.get_classifier()
    out_features = getattr(classifier, "out_features", None)
    if out_features != NUM_CLASSES:
        raise ValueError(
            f"model {config.name!r} produced a head with {out_features} outputs, "
            f"expected {NUM_CLASSES} (from ai.preprocessing.labels.CLASS_CODES)"
        )
    return model


def backbone_blocks(model: nn.Module) -> nn.Sequential | nn.ModuleList:
    """The sequential feature blocks the unfreezing schedule counts over."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError(
            f"{type(model).__name__} exposes no `blocks`; the progressive "
            "unfreezing schedule needs an indexable sequence of feature blocks"
        )
    return blocks


def unfrozen_modules(model: nn.Module, unfreeze_blocks: int) -> list[nn.Module]:
    """Modules trainable in a stage: the head, plus the last N blocks.

    `unfreeze_blocks` is checked against the model's real block count, so an
    over-large value fails loudly instead of quietly unfreezing everything.
    """
    blocks = backbone_blocks(model)
    if unfreeze_blocks > len(blocks):
        raise ValueError(
            f"unfreeze_blocks={unfreeze_blocks} exceeds the model's "
            f"{len(blocks)} backbone blocks"
        )

    modules: list[nn.Module] = [model.get_classifier()]
    if unfreeze_blocks > 0:
        modules.extend(blocks[len(blocks) - unfreeze_blocks :])
        for name in FINAL_MODULE_NAMES:
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
    return modules


def apply_stage(model: nn.Module, stage: StageConfig) -> None:
    """Set `requires_grad` for one stage. Freeze everything, then re-open."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for module in unfrozen_modules(model, stage.unfreeze_blocks):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def set_train_mode(model: nn.Module, stage: StageConfig) -> None:
    """Put the model in training mode, keeping frozen BatchNorms in eval.

    Must be called at the start of every training epoch: `model.train()`
    switches *all* submodules back to train mode, so the frozen-BatchNorm
    exception has to be re-applied each time.
    """
    model.train()

    trainable_norms = {
        id(sub)
        for module in unfrozen_modules(model, stage.unfreeze_blocks)
        for sub in module.modules()
        if isinstance(sub, _BatchNorm)
    }
    for module in model.modules():
        if isinstance(module, _BatchNorm) and id(module) not in trainable_norms:
            module.eval()


def trainable_parameters(model: nn.Module) -> Iterator[torch.nn.Parameter]:
    """The parameters an optimizer should own for the current stage."""
    return (p for p in model.parameters() if p.requires_grad)


def count_trainable_parameters(model: nn.Module) -> int:
    """How many parameters the current stage will update. Reported per stage."""
    return sum(p.numel() for p in trainable_parameters(model))


def input_size_note(model: nn.Module, image_size: int) -> str | None:
    """Describe a mismatch between the configured crop and the model's native
    input, or None when they agree. Reported, never enforced: a different crop
    is a legitimate choice, a silent one is not.
    """
    default_cfg = getattr(model, "default_cfg", None) or {}
    input_size = default_cfg.get("input_size")
    if not input_size or input_size[-1] == image_size:
        return None
    architecture = default_cfg.get("architecture", "model")
    return (
        f"dataset image_size={image_size} differs from {architecture} "
        f"native input {input_size[-1]}"
    )
