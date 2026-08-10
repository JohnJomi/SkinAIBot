"""Model development: EfficientNet-B4 fine-tuning (Phase 2.2).

Public entry points:

- ``load_training_config``  read the reproducible training configuration
- ``build_model``           EfficientNet-B4 with a head sized to the classes

The two runnable steps are CLI modules, and are deliberately not imported here:
importing them would put them in ``sys.modules`` before ``python -m`` executes
them, which runpy warns about. Mirrors Phase 2.1, where ``prepare`` is likewise
not re-exported.

    python -m ai.training.train      fine-tune over train/val
    python -m ai.training.evaluate   score a checkpoint on the test split
"""

from ai.training.config import TrainingConfig, load_training_config
from ai.training.model import build_model

__all__ = ["TrainingConfig", "load_training_config", "build_model"]
