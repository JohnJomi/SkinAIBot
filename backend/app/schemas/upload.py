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
    # Absolute URL the stored image is served from. `/api/v1/analyze` requires
    # an `image_url` its AI service can fetch server-side, so the client is
    # given a usable URL rather than having to construct one it cannot know.
    image_url: str

    model_config = {"from_attributes": True}
