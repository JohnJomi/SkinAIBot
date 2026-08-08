import httpx
import pytest
from backend.app.ai.client import AIClient
from backend.app.ai.schemas import AIAnalyzeRequest, AIChatRequest

@pytest.mark.asyncio
async def test_backend_ai_client_analyze():
    client = AIClient()
    req = AIAnalyzeRequest(analysis_id="123", image_url="http://example.com/image.jpg")
    
    try:
        resp = await client.analyze(req)
        assert resp.analysis_id == "123"
        assert resp.status == "completed"
        assert len(resp.predictions) > 0
    except Exception as e:
        pytest.fail(f"Backend AI client analyze failed: {e}")

@pytest.mark.asyncio
async def test_backend_ai_client_chat():
    client = AIClient()
    req = AIChatRequest(session_id="session1", message="hello")
    
    try:
        resp = await client.chat(req)
        assert resp.response is not None
    except Exception as e:
        pytest.fail(f"Backend AI client chat failed: {e}")
