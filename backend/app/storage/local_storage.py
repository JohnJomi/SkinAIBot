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

    def list_stored_files(self) -> set[str]:
        """Return the set of filenames in the upload directory.

        Only includes files whose extension matches a known upload format,
        so unrelated files (e.g. .gitkeep) are never touched.
        """
        known_extensions = set(FORMAT_EXTENSIONS.values())
        return {
            p.name
            for p in self.upload_dir.iterdir()
            if p.is_file() and p.suffix in known_extensions
        }

    def path_for(self, stored_name: str) -> Path:
        """Filesystem path of a stored file, for serving it back."""
        return self._stored_path(stored_name)

    def url_for(self, stored_name: str) -> str:
        """Path at which a stored file is served. See the uploads router."""
        return f"/api/v1/uploads/{stored_name}"

    def internal_url_for(self, stored_name: str) -> str:
        """Absolute URL for server-side fetching, e.g. by the AI service.

        `/api/v1/analyze` takes an `image_url` the AI service fetches itself, so
        a relative path or a browser-only host is useless to it. Inside Compose
        this resolves to the backend's service name.
        """
        base = get_settings().internal_base_url.rstrip("/")
        return f"{base}{self.url_for(stored_name)}"

    def browser_url_for(self, stored_name: str) -> str:
        """Absolute URL the user's browser can load.

        Separate from `internal_url_for` because the browser sits outside the
        Compose network and cannot resolve a service name. Same path, different
        host - never the filesystem location.
        """
        base = get_settings().browser_base_url.rstrip("/")
        return f"{base}{self.url_for(stored_name)}"
