from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class AIAnalyzeRequest(BaseModel):
    analysis_id: str
    image_url: HttpUrl
    model_version: Optional[str] = "v1"

class AIPrediction(BaseModel):
    label: str
    confidence: float

class AIAnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    model_version: str
    predictions: List[AIPrediction]
    confidence_status: str
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
