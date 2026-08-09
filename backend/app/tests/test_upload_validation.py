import io

import pytest
from PIL import Image

from app.exceptions import InvalidFileTypeError
from app.services.upload_service import detect_image_format
from app.storage.local_storage import FORMAT_EXTENSIONS, LocalFileStorage


def make_image_bytes(image_format: str, size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_detect_image_format_accepts_supported_images(image_format):
    assert detect_image_format(make_image_bytes(image_format)) == image_format


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("html", b"<html><body>not an image</body></html>"),
        ("plain_text", b"just some text"),
        ("empty", b""),
        ("png_magic_bytes_only", b"\x89PNG\r\n\x1a\n" + b"garbage" * 8),
        ("elf_binary", b"\x7fELF" + b"\x00" * 64),
    ],
)
def test_detect_image_format_rejects_non_images(label, content):
    """Content-Type is attacker-controlled; only the bytes decide."""
    with pytest.raises(InvalidFileTypeError):
        detect_image_format(content)


def test_detect_image_format_rejects_unsupported_image_format():
    """A real image in a format outside the allow-list is still rejected."""
    with pytest.raises(InvalidFileTypeError):
        detect_image_format(make_image_bytes("BMP"))


def test_stored_extension_comes_from_detected_format_not_filename(tmp_path, monkeypatch):
    """A .php/.jpg filename must never influence the stored extension."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        storage = LocalFileStorage()
        stored_name = storage.save(make_image_bytes("PNG"), "PNG")
    finally:
        get_settings.cache_clear()

    assert stored_name.endswith(".png")
    assert (tmp_path / stored_name).exists()


def test_storage_rejects_unknown_format(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        storage = LocalFileStorage()
        with pytest.raises(ValueError):
            storage.allocate_name("PHP")
    finally:
        get_settings.cache_clear()


def test_format_extension_map_covers_supported_formats():
    assert set(FORMAT_EXTENSIONS) == {"JPEG", "PNG", "WEBP"}
