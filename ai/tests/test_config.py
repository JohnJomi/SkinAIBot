"""Configuration validation."""

import pytest
import yaml
from pydantic import ValidationError

from ai.preprocessing.config import load_config


def _config_with_seed(config_path, tmp_path, seed):
    """The suite's config YAML, rewritten with a different seed."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["seed"] = seed
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_negative_seed_rejected(config_path, tmp_path):
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_config(_config_with_seed(config_path, tmp_path, -1))


@pytest.mark.parametrize("seed", [0, 42])
def test_zero_and_positive_seeds_accepted(config_path, tmp_path, seed):
    assert load_config(_config_with_seed(config_path, tmp_path, seed)).seed == seed
