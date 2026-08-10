"""Canonical class mapping for the HAM10000 skin lesion dataset.

This module is the single source of truth for label order. The index a class
receives here is the index the model's output vector will use, so the tuple
below must never be reordered once a model has been trained against it.
"""

from collections.abc import Iterable

# Ordered alphabetically by dataset code. Alphabetical order is arbitrary but
# stable: it does not depend on the class frequencies of whatever copy of the
# dataset happens to be on disk.
CLASS_CODES: tuple[str, ...] = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)

# Human-readable names, used for reporting. Not consumed by the pipeline.
CLASS_NAMES: dict[str, str] = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

NUM_CLASSES: int = len(CLASS_CODES)

_CLASS_TO_INDEX: dict[str, int] = {code: i for i, code in enumerate(CLASS_CODES)}


def class_to_index() -> dict[str, int]:
    """Return a copy of the class code -> model output index mapping."""
    return dict(_CLASS_TO_INDEX)


def index_to_class() -> tuple[str, ...]:
    """Return the class codes in model output order."""
    return CLASS_CODES


def encode(code: str) -> int:
    """Map a single dataset class code to its model output index."""
    try:
        return _CLASS_TO_INDEX[code]
    except KeyError:
        raise ValueError(
            f"Unknown class code {code!r}; expected one of {list(CLASS_CODES)}"
        ) from None


def validate_codes(codes: Iterable[str]) -> None:
    """Raise if `codes` contains anything outside the canonical class set.

    Called against the raw metadata so that a dataset revision introducing a new
    diagnosis fails immediately rather than being silently dropped.
    """
    unknown = sorted({code for code in codes} - set(CLASS_CODES))
    if unknown:
        raise ValueError(
            f"Metadata contains unknown class codes {unknown}; "
            f"expected only {list(CLASS_CODES)}"
        )
