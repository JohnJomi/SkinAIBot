"""Images are loaded exactly as the dataset loads them, or rejected clearly."""

import numpy as np
import pytest
from PIL import Image

from ai.inference.images import InvalidImageError, describe, load_rgb


@pytest.fixture
def jpeg(tmp_path):
    path = tmp_path / "lesion.jpg"
    pixels = np.random.default_rng(0).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="JPEG")
    return path


def test_rgb_image_loads_unchanged_in_mode(jpeg):
    image = load_rgb(jpeg)

    assert image.mode == "RGB"
    assert image.size == (30, 40)


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("L", ".png"), ("RGBA", ".png"), ("P", ".png"), ("CMYK", ".tiff")],
)
def test_other_modes_are_converted_to_rgb(tmp_path, mode, suffix):
    # A greyscale or RGBA file would otherwise reach the transform with the
    # wrong channel count and produce a silently wrong tensor. The suffix
    # varies only because PNG cannot store CMYK.
    path = tmp_path / f"image{suffix}"
    Image.new(mode, (20, 20), color=0).save(path)

    assert load_rgb(path).mode == "RGB"


def test_pil_image_input_is_accepted():
    assert load_rgb(Image.new("L", (8, 8))).mode == "RGB"


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_rgb(tmp_path / "absent.jpg")


def test_directory_is_rejected(tmp_path):
    with pytest.raises(InvalidImageError, match="not a file"):
        load_rgb(tmp_path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")

    with pytest.raises(InvalidImageError, match="is empty"):
        load_rgb(path)


def test_non_image_bytes_are_rejected(tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("this is not a JPEG", encoding="utf-8")

    with pytest.raises(InvalidImageError, match="not a recognised image format"):
        load_rgb(path)


def test_truncated_image_is_rejected(tmp_path, jpeg):
    # A valid header with the pixel data cut off: only forcing the decode
    # catches this, because Image.open is lazy.
    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(jpeg.read_bytes()[:80])

    with pytest.raises(InvalidImageError):
        load_rgb(truncated)


def test_describe_identifies_the_source(jpeg):
    assert describe(jpeg) == str(jpeg)
    assert describe(Image.new("RGB", (4, 4))) == "<image>"
