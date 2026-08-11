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
    # Two URLs for the same stored image, named for who can actually fetch
    # each. Collapsing them into one `image_url` is what made it ambiguous:
    # the value the AI service needs is not loadable by the browser.
    #
    # Pass this to `/api/v1/analyze`; the AI service fetches it server-side.
    analysis_image_url: str
    # Use this in an <img>; the browser can resolve this host.
    display_image_url: str

    model_config = {"from_attributes": True}
