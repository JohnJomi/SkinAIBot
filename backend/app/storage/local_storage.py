"""Local filesystem storage for uploaded images."""

import uuid
from pathlib import Path

from ..config import get_settings


class LocalFileStorage:
    def __init__(self):
        self.upload_dir = Path(get_settings().upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix
        stored_name = f"{uuid.uuid4()}{suffix}"
        (self.upload_dir / stored_name).write_bytes(content)
        return stored_name

    def url_for(self, stored_name: str) -> str:
        return f"/uploads/{stored_name}"
