"""Loading a single image for inference.

Deliberately mirrors `HAM10000Dataset.__getitem__`: open, then convert to RGB.
The conversion is not cosmetic - a greyscale or RGBA file would otherwise reach
the transform with the wrong channel count and produce a silently wrong tensor.

Failure modes are separated on purpose. A path that does not exist is a
`FileNotFoundError`, because that is what the caller can act on; anything that
exists but cannot be decoded is an `InvalidImageError`, so a corrupt upload is
never mistaken for a missing file.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

type ImageSource = str | Path | Image.Image


class InvalidImageError(ValueError):
    """The input exists but cannot be read as an image."""


def load_rgb(source: ImageSource) -> Image.Image:
    """Return `source` as an RGB image, whatever it started as."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"image {path} not found")
    if not path.is_file():
        raise InvalidImageError(f"{path} is not a file")
    if path.stat().st_size == 0:
        raise InvalidImageError(f"{path} is empty")

    try:
        with Image.open(path) as image:
            # Force a decode inside the context: Image.open is lazy, so a
            # truncated file would otherwise fail much later, out of context.
            image.load()
            return image.convert("RGB")
    except UnidentifiedImageError:
        raise InvalidImageError(
            f"{path} is not a recognised image format"
        ) from None
    except OSError as error:
        raise InvalidImageError(f"{path} could not be decoded: {error}") from None


def describe(source: ImageSource) -> str:
    """A stable identifier for a source, for reports and error messages."""
    if isinstance(source, Image.Image):
        return "<image>"
    return str(source)
