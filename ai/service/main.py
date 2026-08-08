from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

app = FastAPI(title="SkinAIBot Mock AI Service", version="0.1.0")

class AnalyzeRequest(BaseModel):
    analysis_id: str
    image_url: HttpUrl
    model_version: Optional[str] = "v1"

class Prediction(BaseModel):
    label: str
    confidence: float

class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: str
    model_version: str
    predictions: List[Prediction]
    confidence_status: str
    explanation: str
    recommendation: str
    disclaimer: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    analysis_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    disclaimer: str

@app.post("/ai/analyze", response_model=AnalyzeResponse)
def analyze_image(req: AnalyzeRequest):
    return AnalyzeResponse(
        analysis_id=req.analysis_id,
        status="completed",
        model_version="mock-efficientnet-v1",
        predictions=[
            Prediction(label="melanoma", confidence=0.85),
            Prediction(label="nevus", confidence=0.10),
            Prediction(label="seborrheic_keratosis", confidence=0.05),
        ],
        confidence_status="high",
        explanation="The model detected features highly indicative of melanoma. [MOCK]",
        recommendation="Immediate dermatological consultation is advised. [MOCK]",
        disclaimer="This is a mock AI response for development purposes."
    )

@app.post("/ai/chat", response_model=ChatResponse)
def chat_ai(req: ChatRequest):
    return ChatResponse(
        response=f"This is a mock chat response to your message: '{req.message}'.",
        disclaimer="This is a mock AI response for development purposes."
    )

@app.get("/ai/health")
def health():
    return {"status": "healthy", "service": "Mock AI"}
