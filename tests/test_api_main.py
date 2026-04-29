"""Tests for src/api/main.py — FastAPI endpoint tests using TestClient."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.chat import ChatHandler, _NOT_CANVAS_ANSWER
from src.api.models import ChatResponse, SourceRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas_response(answer: str = "Canvas 답변입니다.") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        domain="canvas",
        sources=[SourceRef(
            title="Submit Assignment",
            source_url="https://community.instructure.com/en/kb/articles/661210",
            score=0.91,
        )],
        matched_keywords=["assignment"],
    )


def _make_general_response() -> ChatResponse:
    return ChatResponse(
        answer=_NOT_CANVAS_ANSWER,
        domain="general",
        sources=[],
        matched_keywords=[],
    )


def _make_mock_handler(response: ChatResponse) -> MagicMock:
    handler = MagicMock(spec=ChatHandler)
    handler.handle.return_value = response
    return handler


# ---------------------------------------------------------------------------
# App fixture with handler injected
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_canvas():
    """TestClient with a mock handler that returns a Canvas response.

    TestClient is used WITHOUT the context-manager `with` to skip lifespan,
    then _handler is patched directly so the lifespan never overwrites it.
    """
    from src.api import main as api_main
    mock_handler = _make_mock_handler(_make_canvas_response())
    client = TestClient(api_main.app, raise_server_exceptions=True)
    with patch.object(api_main, "_handler", mock_handler):
        yield client, mock_handler


@pytest.fixture()
def client_general():
    """TestClient with a mock handler that returns a general response."""
    from src.api import main as api_main
    mock_handler = _make_mock_handler(_make_general_response())
    client = TestClient(api_main.app, raise_server_exceptions=True)
    with patch.object(api_main, "_handler", mock_handler):
        yield client


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client_canvas):
        client, _ = client_canvas
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body_has_status_ok(self, client_canvas):
        client, _ = client_canvas
        body = client.get("/health").json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /chat — basic
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_returns_200_on_canvas_query(self, client_canvas):
        client, _ = client_canvas
        resp = client.post("/chat", json={"query": "How do I submit an assignment?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, client_canvas):
        client, _ = client_canvas
        body = client.post("/chat", json={"query": "assignment submission"}).json()
        assert "answer" in body
        assert body["answer"] == "Canvas 답변입니다."

    def test_response_has_domain(self, client_canvas):
        client, _ = client_canvas
        body = client.post("/chat", json={"query": "assignment"}).json()
        assert body["domain"] == "canvas"

    def test_response_has_sources(self, client_canvas):
        client, _ = client_canvas
        body = client.post("/chat", json={"query": "assignment"}).json()
        assert isinstance(body["sources"], list)
        assert len(body["sources"]) == 1
        assert "source_url" in body["sources"][0]

    def test_response_has_matched_keywords(self, client_canvas):
        client, _ = client_canvas
        body = client.post("/chat", json={"query": "assignment"}).json()
        assert isinstance(body["matched_keywords"], list)

    def test_handler_called_with_correct_query(self, client_canvas):
        client, mock_handler = client_canvas
        client.post("/chat", json={"query": "quiz creation"})
        called_request = mock_handler.handle.call_args[0][0]
        assert called_request.query == "quiz creation"

    def test_role_passed_to_handler(self, client_canvas):
        client, mock_handler = client_canvas
        client.post("/chat", json={"query": "assignment", "role": "student"})
        called_request = mock_handler.handle.call_args[0][0]
        assert called_request.role == "student"

    def test_force_domain_passed_to_handler(self, client_canvas):
        client, mock_handler = client_canvas
        client.post("/chat", json={"query": "anything", "force_domain": "canvas"})
        called_request = mock_handler.handle.call_args[0][0]
        assert called_request.force_domain == "canvas"

    def test_top_k_passed_to_handler(self, client_canvas):
        client, mock_handler = client_canvas
        client.post("/chat", json={"query": "assignment", "top_k": 3})
        called_request = mock_handler.handle.call_args[0][0]
        assert called_request.top_k == 3

    def test_general_query_returns_redirect(self, client_general):
        body = client_general.post("/chat", json={"query": "오늘 날씨"}).json()
        assert body["answer"] == _NOT_CANVAS_ANSWER
        assert body["domain"] == "general"

    def test_empty_query_returns_422(self, client_canvas):
        client, _ = client_canvas
        resp = client.post("/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client_canvas):
        client, _ = client_canvas
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_top_k_too_large_returns_422(self, client_canvas):
        client, _ = client_canvas
        resp = client.post("/chat", json={"query": "assignment", "top_k": 99})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_no_auth_when_api_key_not_set(self, client_canvas):
        """Without API_KEY env var, no auth required."""
        client, _ = client_canvas
        resp = client.post("/chat", json={"query": "assignment"})
        assert resp.status_code == 200

    def test_auth_required_when_api_key_set(self, client_canvas):
        """When API_KEY is set, missing header → 401."""
        import src.api.main as api_main
        client, _ = client_canvas
        original = api_main._API_KEY
        try:
            api_main._API_KEY = "secret-key"
            resp = client.post("/chat", json={"query": "assignment"})
            assert resp.status_code == 401
        finally:
            api_main._API_KEY = original

    def test_valid_api_key_accepted(self, client_canvas):
        import src.api.main as api_main
        client, _ = client_canvas
        original = api_main._API_KEY
        try:
            api_main._API_KEY = "secret-key"
            resp = client.post(
                "/chat",
                json={"query": "assignment"},
                headers={"X-API-Key": "secret-key"},
            )
            assert resp.status_code == 200
        finally:
            api_main._API_KEY = original

    def test_wrong_api_key_rejected(self, client_canvas):
        import src.api.main as api_main
        client, _ = client_canvas
        original = api_main._API_KEY
        try:
            api_main._API_KEY = "secret-key"
            resp = client.post(
                "/chat",
                json={"query": "assignment"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 401
        finally:
            api_main._API_KEY = original
