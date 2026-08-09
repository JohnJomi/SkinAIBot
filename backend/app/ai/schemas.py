from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class AIAnalyzeRequest(BaseModel):
    analysis_id: str
    image_url: HttpUrl
    model_version: Optional[str] = "v1"

class AIPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

class AIAnalyzeResponse(BaseModel):
    analysis_id: str
    status: Literal["completed", "failed", "processing"]
    model_version: str
    predictions: List[AIPrediction]
    confidence_status: Literal["high", "moderate", "low"]
    explanation: str
    recommendation: str
    disclaimer: str

class AIChatRequest(BaseModel):
    session_id: str
    message: str
    analysis_id: Optional[str] = None

class AIChatResponse(BaseModel):
    response: str
    disclaimer: str
