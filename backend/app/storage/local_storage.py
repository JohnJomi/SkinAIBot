"""Local filesystem storage for uploaded images."""

import os
import uuid
from pathlib import Path

from ..config import get_settings

# The stored extension is derived only from the format detected in the file's
# own bytes. Never from the client-supplied filename, which an attacker
# controls and could use to write e.g. ".php" or a traversal sequence.
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


class LocalFileStorage:
    def __init__(self):
        self.upload_dir = Path(get_settings().upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _stored_path(self, stored_name: str) -> Path:
        return self.upload_dir / stored_name

    def allocate_name(self, image_format: str) -> str:
        try:
            suffix = FORMAT_EXTENSIONS[image_format]
        except KeyError:
            raise ValueError(f"Unsupported image format '{image_format}'.") from None
        return f"{uuid.uuid4()}{suffix}"

    def save(self, content: bytes, image_format: str) -> str:
        stored_name = self.allocate_name(image_format)
        self._stored_path(stored_name).write_bytes(content)
        return stored_name

    def new_temp_path(self) -> Path:
        """A scratch path inside the upload dir, so promote() is an atomic rename."""
        return self.upload_dir / f".incoming-{uuid.uuid4()}.tmp"

    def promote(self, temp_path: Path, image_format: str) -> str:
        """Move a fully written temp file to its final name."""
        stored_name = self.allocate_name(image_format)
        os.replace(temp_path, self._stored_path(stored_name))
        return stored_name

    def discard(self, temp_path: Path) -> None:
        """Remove a temp file. Safe to call when it was never created."""
        Path(temp_path).unlink(missing_ok=True)

    def delete(self, stored_name: str) -> None:
        """Remove a stored file. Safe to call when the file is already gone."""
        self._stored_path(stored_name).unlink(missing_ok=True)

    def url_for(self, stored_name: str) -> str:
        return f"/uploads/{stored_name}"
