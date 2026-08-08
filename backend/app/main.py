from fastapi import FastAPI

app = FastAPI(
    title="Skin Disease Diagnosis API",
    description="Backend API for the Skin Disease Diagnosis AI Platform.",
    version="0.1.0",
)

from .ai.schemas import AIAnalyzeRequest, AIAnalyzeResponse, AIChatRequest, AIChatResponse
from .ai import get_ai_client


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "Skin Disease Diagnosis API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/api/v1/analyze", response_model=AIAnalyzeResponse)
async def analyze_image_endpoint(req: AIAnalyzeRequest):
    client = get_ai_client()
    return await client.analyze(req)

@app.post("/api/v1/chat", response_model=AIChatResponse)
async def chat_endpoint(req: AIChatRequest):
    client = get_ai_client()
    return await client.chat(req)
