"""Pydantic schemas for uploads."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    content_type: str
    size_bytes: int
    created_at: datetime

    model_config = {"from_attributes": True}
