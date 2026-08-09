import json
import os

import httpx
from pydantic import ValidationError

from ..exceptions import AIServiceUnavailableError, AIUpstreamResponseError
from .schemas import AIAnalyzeRequest, AIAnalyzeResponse, AIChatRequest, AIChatResponse

AI_REQUEST_TIMEOUT = 30.0

class AIClient:
    def __init__(self):
        self.base_url = os.getenv("AI_SERVICE_URL", "http://localhost:8001").rstrip("/")

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
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise AIUpstreamResponseError(
                f"AI service returned a malformed JSON response for {path}."
            ) from exc

    def _validate(self, path: str, model: type, data: dict):
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise AIUpstreamResponseError(
                f"AI service returned a response that does not conform to the contract for {path}."
            ) from exc

    async def analyze(self, req: AIAnalyzeRequest) -> AIAnalyzeResponse:
        data = await self._post_json("/ai/analyze", req.model_dump(mode='json'))
        return self._validate("/ai/analyze", AIAnalyzeResponse, data)

    async def chat(self, req: AIChatRequest) -> AIChatResponse:
        data = await self._post_json("/ai/chat", req.model_dump(mode='json'))
        return self._validate("/ai/chat", AIChatResponse, data)
