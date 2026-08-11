"""Image inference (Phase 2.4).

Public entry points:

- ``Predictor``   run a validated Phase 2.2 checkpoint over images
- ``Prediction``  the structured result

The CLI is a module rather than an export, mirroring Phase 2.1's ``prepare``
and Phase 2.2's ``train``:

    python -m ai.inference.predict IMAGE [IMAGE ...]

This package is a library and a CLI. It deliberately contains no HTTP layer:
``ai/service`` is a separate, Phase 3 concern.
"""

from ai.inference.images import InvalidImageError
from ai.inference.predictor import Predictor
from ai.inference.result import Prediction

__all__ = ["Predictor", "Prediction", "InvalidImageError"]
