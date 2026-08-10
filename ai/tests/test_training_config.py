"""The training configuration validates, and stages are pure data."""

import pytest
import yaml
from pydantic import ValidationError

from ai.preprocessing.labels import NUM_CLASSES
from ai.training.config import (
    DEFAULT_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
)


@pytest.fixture(scope="module")
def shipped_config() -> TrainingConfig:
    """The configuration actually committed to ai/configs/training.yaml."""
    return load_training_config(DEFAULT_CONFIG_PATH)


def _payload() -> dict:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_shipped_defaults_match_the_approved_plan(shipped_config):
    assert shipped_config.imbalance_strategy == "weighted_loss"
    assert shipped_config.monitor == "macro_recall"
    # A full weight tag, not a bare architecture: timm's default tag for an
    # architecture can change between releases.
    assert shipped_config.model.name == "tf_efficientnet_b4.aa_in1k"
    assert shipped_config.model.pretrained is True


def test_shipped_stages_are_head_then_finetune(shipped_config):
    head, finetune = shipped_config.stages

    assert (head.name, head.epochs, head.lr, head.unfreeze_blocks) == (
        "head",
        8,
        1.0e-3,
        0,
    )
    assert (finetune.name, finetune.epochs, finetune.lr, finetune.unfreeze_blocks) == (
        "finetune",
        12,
        1.0e-4,
        2,
    )
    assert shipped_config.total_epochs == 20


def test_relative_paths_resolve_against_the_repository_root(shipped_config):
    assert shipped_config.checkpoint_dir.is_absolute()
    assert shipped_config.report_dir.is_absolute()
    assert shipped_config.checkpoint_dir.name == "checkpoints"


def test_a_third_stage_needs_no_code_change(tmp_path):
    # The whole point of stages-as-data: this must load and be iterable with
    # nothing in Python knowing how many stages exist.
    payload = _payload()
    payload["stages"].append(
        {
            "name": "deep_finetune",
            "epochs": 4,
            "lr": 3.0e-5,
            "weight_decay": 1.0e-4,
            "unfreeze_blocks": 4,
            "scheduler": "cosine",
        }
    )
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_training_config(path)
    assert len(config.stages) == 3
    assert config.stages[-1].unfreeze_blocks == 4
    assert config.total_epochs == 24


def test_duplicate_stage_names_rejected():
    payload = _payload()
    payload["stages"][1]["name"] = payload["stages"][0]["name"]

    with pytest.raises(ValidationError, match="stage names must be unique"):
        TrainingConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("imbalance_strategy", "both"),
        ("monitor", "loss"),
        ("device", "tpu"),
        ("label_smoothing", 1.0),
    ],
)
def test_invalid_top_level_values_rejected(key, value):
    payload = _payload()
    payload[key] = value
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("epochs", 0),
        ("lr", 0.0),
        ("weight_decay", -1.0),
        ("unfreeze_blocks", -1),
        ("scheduler", "step"),
    ],
)
def test_invalid_stage_values_rejected(key, value):
    payload = _payload()
    payload["stages"][0][key] = value
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


@pytest.mark.parametrize(
    "name",
    [
        "tf_efficientnet_b4",  # bare architecture: follows timm's default tag
        "tf_efficientnet_b4.",  # empty tag
        ".aa_in1k",  # empty architecture
        "",
    ],
)
def test_pretrained_without_an_explicit_tag_rejected(name):
    payload = _payload()
    payload["model"]["name"] = name
    payload["model"]["pretrained"] = True

    with pytest.raises(ValidationError, match="no explicit pretrained tag"):
        TrainingConfig.model_validate(payload)


def test_tagged_name_accepted_when_pretrained():
    payload = _payload()
    payload["model"]["name"] = "tf_efficientnet_b4.ns_jft_in1k"

    assert (
        TrainingConfig.model_validate(payload).model.name
        == "tf_efficientnet_b4.ns_jft_in1k"
    )


def test_bare_name_allowed_when_not_pretrained():
    # Nothing is downloaded, so there is no tag to pin.
    payload = _payload()
    payload["model"]["name"] = "tf_efficientnet_b4"
    payload["model"]["pretrained"] = False

    config = TrainingConfig.model_validate(payload)
    assert config.model.name == "tf_efficientnet_b4"
    assert config.model.pretrained is False


def test_shipped_evaluation_block_targets_the_roadmap_metric(shipped_config):
    evaluation = shipped_config.evaluation

    assert evaluation.top_k == 3
    assert evaluation.calibration_bins == 15
    assert evaluation.precision_floor == 0.5
    # docs/ROADMAP.md sets weighted F1 > 0.85 as the project target.
    assert evaluation.targets.as_dict() == {"weighted_f1": 0.85}


def test_evaluation_block_is_optional():
    # Configs written before the evaluation layer must still load.
    payload = _payload()
    del payload["evaluation"]

    config = TrainingConfig.model_validate(payload)
    assert config.evaluation.top_k == 3
    assert config.evaluation.targets.as_dict() == {}


def test_unset_targets_are_not_checked():
    payload = _payload()
    payload["evaluation"]["targets"] = {}

    assert TrainingConfig.model_validate(payload).evaluation.targets.as_dict() == {}


def test_unknown_target_metric_rejected():
    # A typo must fail at config load, not silently check nothing.
    payload = _payload()
    payload["evaluation"]["targets"] = {"f1": 0.85}

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("top_k", 0),
        # Larger than the class count: caught at config load, not downstream.
        ("top_k", NUM_CLASSES + 1),
        ("calibration_bins", 0),
        ("precision_floor", 1.5),
        ("worst_n", -1),
        ("unknown_setting", 1),
    ],
)
def test_invalid_evaluation_values_rejected(key, value):
    payload = _payload()
    payload["evaluation"][key] = value

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


def test_target_outside_the_unit_interval_rejected():
    payload = _payload()
    payload["evaluation"]["targets"]["weighted_f1"] = 1.5

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)


def test_at_least_one_stage_required():
    payload = _payload()
    payload["stages"] = []
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(payload)
