import os

import httpx

from ..exceptions import AIServiceUnavailableError, AIUpstreamResponseError
from .schemas import AIAnalyzeRequest, AIAnalyzeResponse, AIChatRequest, AIChatResponse

AI_REQUEST_TIMEOUT = 30.0

class AIClient:
    def __init__(self):
        self.base_url = os.getenv("AI_SERVICE_URL", "http://localhost:8001")

    async def _post_json(self, path: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT) as client:
                resp = await client.post(f"{self.base_url}{path}", json=payload)
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AIUpstreamResponseError(
                f"AI service returned HTTP {exc.response.status_code} for {path}."
            ) from exc
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            raise AIServiceUnavailableError(
                f"AI service at {self.base_url} is unreachable or timed out."
            ) from exc
        return resp.json()

    async def analyze(self, req: AIAnalyzeRequest) -> AIAnalyzeResponse:
        data = await self._post_json("/ai/analyze", req.model_dump(mode='json'))
        return AIAnalyzeResponse.model_validate(data)

    async def chat(self, req: AIChatRequest) -> AIChatResponse:
        data = await self._post_json("/ai/chat", req.model_dump(mode='json'))
        return AIChatResponse.model_validate(data)
