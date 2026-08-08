import httpx
import os
from .schemas import AIAnalyzeRequest, AIAnalyzeResponse, AIChatRequest, AIChatResponse

class AIClient:
    def __init__(self):
        self.base_url = os.getenv("AI_SERVICE_URL", "http://localhost:8001")

    async def analyze(self, req: AIAnalyzeRequest) -> AIAnalyzeResponse:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/ai/analyze",
                json=req.model_dump(mode='json')
            )
            resp.raise_for_status()
            return AIAnalyzeResponse.model_validate(resp.json())

    async def chat(self, req: AIChatRequest) -> AIChatResponse:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/ai/chat",
                json=req.model_dump(mode='json')
            )
            resp.raise_for_status()
            return AIChatResponse.model_validate(resp.json())
