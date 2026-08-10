"""Head sizing, stage freezing, and the frozen-BatchNorm guarantee."""

import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from ai.preprocessing.labels import NUM_CLASSES
from ai.training.config import ModelConfig, StageConfig
from ai.training.model import (
    apply_stage,
    backbone_blocks,
    build_model,
    count_trainable_parameters,
    input_size_note,
    set_train_mode,
    unfrozen_modules,
)

# EfficientNet is built from many small ops, and multi-threaded CPU dispatch
# thrashes on them: a single B4 forward measures ~30s at 4 threads and ~0.1s at
# 1. The suite runs one sample at a time, so threading buys nothing here.
torch.set_num_threads(1)

STAGE_1 = StageConfig(
    name="head", epochs=1, lr=1e-3, weight_decay=0.0, unfreeze_blocks=0
)
STAGE_2 = StageConfig(
    name="finetune", epochs=1, lr=1e-4, weight_decay=0.0, unfreeze_blocks=2
)


@pytest.fixture(scope="module")
def model():
    """The real EfficientNet-B4, with random weights.

    pretrained=False keeps the suite offline: no weight download, and the
    architecture is what these tests are about.
    """
    return build_model(
        ModelConfig(name="tf_efficientnet_b4.aa_in1k", pretrained=False, drop_rate=0.4)
    )


def _backbone_parameters(model):
    classifier_ids = {id(p) for p in model.get_classifier().parameters()}
    return [p for p in model.parameters() if id(p) not in classifier_ids]


def test_head_is_sized_from_the_canonical_label_mapping(model):
    # Derived from CLASS_CODES, never a literal 7.
    assert model.get_classifier().out_features == NUM_CLASSES


def test_forward_produces_one_logit_per_class(model):
    logits = model(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, NUM_CLASSES)

    probabilities = torch.softmax(logits, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)


def test_stage_1_freezes_the_backbone_and_keeps_the_classifier_trainable(model):
    apply_stage(model, STAGE_1)

    assert all(not p.requires_grad for p in _backbone_parameters(model))
    assert all(p.requires_grad for p in model.get_classifier().parameters())
    assert count_trainable_parameters(model) == sum(
        p.numel() for p in model.get_classifier().parameters()
    )


def test_stage_1_keeps_frozen_batchnorms_in_eval_mode(model):
    apply_stage(model, STAGE_1)
    set_train_mode(model, STAGE_1)

    norms = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    assert norms, "fixture must contain BatchNorm layers"
    # requires_grad=False does not stop running statistics updating; eval mode
    # is what does, and it is the thing under test here.
    assert all(not norm.training for norm in norms)


def test_stage_1_forward_does_not_move_running_statistics(model):
    apply_stage(model, STAGE_1)
    set_train_mode(model, STAGE_1)

    norm = next(m for m in model.modules() if isinstance(m, _BatchNorm))
    before = norm.running_mean.clone()

    model(torch.randn(2, 3, 64, 64))

    # Bit-identical: a single train-mode BatchNorm would shift this.
    assert torch.equal(norm.running_mean, before)


def test_stage_2_unfreezes_the_last_blocks_and_returns_them_to_train_mode(model):
    blocks = backbone_blocks(model)
    apply_stage(model, STAGE_2)
    set_train_mode(model, STAGE_2)

    unfrozen = blocks[len(blocks) - STAGE_2.unfreeze_blocks :]
    for block in unfrozen:
        assert all(p.requires_grad for p in block.parameters())
        for norm in (m for m in block.modules() if isinstance(m, _BatchNorm)):
            assert norm.training

    still_frozen = blocks[: len(blocks) - STAGE_2.unfreeze_blocks]
    for block in still_frozen:
        assert all(not p.requires_grad for p in block.parameters())
        for norm in (m for m in block.modules() if isinstance(m, _BatchNorm)):
            assert not norm.training


def test_stage_2_also_unfreezes_the_final_projection_and_head(model):
    apply_stage(model, STAGE_2)
    modules = unfrozen_modules(model, STAGE_2.unfreeze_blocks)

    assert model.get_classifier() in modules
    assert model.conv_head in modules
    assert model.bn2 in modules
    assert count_trainable_parameters(model) > sum(
        p.numel() for p in model.get_classifier().parameters()
    )


def test_stage_2_trains_more_parameters_than_stage_1(model):
    apply_stage(model, STAGE_1)
    stage_1_trainable = count_trainable_parameters(model)

    apply_stage(model, STAGE_2)
    assert count_trainable_parameters(model) > stage_1_trainable


def test_unfreezing_more_blocks_than_exist_is_rejected(model):
    too_many = len(backbone_blocks(model)) + 1
    with pytest.raises(ValueError, match="exceeds the model's"):
        unfrozen_modules(model, too_many)


def test_input_size_note_reports_only_a_real_mismatch(model):
    native = model.default_cfg["input_size"][-1]
    assert input_size_note(model, native) is None
    assert "differs from" in input_size_note(model, native // 2)
