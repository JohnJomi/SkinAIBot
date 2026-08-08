import httpx
import pytest
from backend.app.ai.client import AIClient
from backend.app.ai.schemas import (
    AIAnalyzeRequest,
    AIAnalyzeResponse,
    AIChatRequest,
    AIChatResponse,
)
from backend.app.exceptions import AIServiceUnavailableError, AIUpstreamResponseError
from pydantic import ValidationError

VALID_ANALYZE_RESPONSE = {
    "analysis_id": "123",
    "status": "completed",
    "model_version": "mock-efficientnet-v1",
    "predictions": [{"label": "melanoma", "confidence": 0.85}],
    "confidence_status": "high",
    "explanation": "mock explanation",
    "recommendation": "mock recommendation",
    "disclaimer": "mock response for testing",
}

VALID_CHAT_RESPONSE = {
    "response": "mock chat reply",
    "disclaimer": "mock response for testing",
}


@pytest.fixture
def mock_transport_client(monkeypatch):
    """Replace httpx.AsyncClient with a MockTransport-backed client (no network I/O)."""
    monkeypatch.setenv("AI_SERVICE_URL", "http://ai-service:8001")
    original_async_client = httpx.AsyncClient

    def patch(handler):
        def factory(*args, **kwargs):
            return original_async_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return patch


@pytest.mark.asyncio
async def test_analyze_returns_validated_response(mock_transport_client):
    requested_url = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested_url["value"] = str(request.url)
        return httpx.Response(200, json=VALID_ANALYZE_RESPONSE)

    mock_transport_client(handler)

    client = AIClient()
    resp = await client.analyze(
        AIAnalyzeRequest(analysis_id="123", image_url="http://example.com/image.jpg")
    )

    assert isinstance(resp, AIAnalyzeResponse)
    assert requested_url["value"] == "http://ai-service:8001/ai/analyze"
    assert resp.analysis_id == "123"
    assert resp.status == "completed"
    assert resp.predictions[0].label == "melanoma"
    assert resp.predictions[0].confidence == 0.85
    assert resp.confidence_status == "high"
    assert resp.disclaimer == "mock response for testing"


@pytest.mark.asyncio
async def test_analyze_rejects_malformed_response(mock_transport_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"analysis_id": "123"})

    mock_transport_client(handler)

    client = AIClient()
    with pytest.raises(ValidationError):
        await client.analyze(
            AIAnalyzeRequest(analysis_id="123", image_url="http://example.com/image.jpg")
        )


@pytest.mark.asyncio
async def test_analyze_raises_on_upstream_http_error(mock_transport_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "upstream failure"})

    mock_transport_client(handler)

    client = AIClient()
    with pytest.raises(AIUpstreamResponseError) as exc_info:
        await client.analyze(
            AIAnalyzeRequest(analysis_id="123", image_url="http://example.com/image.jpg")
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == "AI_UPSTREAM_ERROR"


@pytest.mark.asyncio
async def test_analyze_raises_unavailable_on_connect_error(mock_transport_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    mock_transport_client(handler)

    client = AIClient()
    with pytest.raises(AIServiceUnavailableError) as exc_info:
        await client.analyze(
            AIAnalyzeRequest(analysis_id="123", image_url="http://example.com/image.jpg")
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "AI_SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_chat_returns_validated_response(mock_transport_client):
    requested_url = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requested_url["value"] = str(request.url)
        return httpx.Response(200, json=VALID_CHAT_RESPONSE)

    mock_transport_client(handler)

    client = AIClient()
    resp = await client.chat(AIChatRequest(session_id="session1", message="hello"))

    assert isinstance(resp, AIChatResponse)
    assert requested_url["value"] == "http://ai-service:8001/ai/chat"
    assert resp.response == "mock chat reply"
    assert resp.disclaimer == "mock response for testing"
