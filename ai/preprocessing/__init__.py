"""Dataset -> model-ready pipeline (Phase 2.1).

Public entry points:

- ``load_config``      read the reproducible data configuration
- ``assign_splits``    lesion-grouped, class-stratified train/val/test split
- ``build_dataloaders`` construct the three DataLoaders from split manifests
"""

from ai.preprocessing.config import DataConfig, load_config
from ai.preprocessing.datamodule import build_dataloaders
from ai.preprocessing.splits import assign_splits

__all__ = ["DataConfig", "load_config", "assign_splits", "build_dataloaders"]
