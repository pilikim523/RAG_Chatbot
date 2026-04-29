"""
FastAPI application — Canvas RAG Chatbot API.

Endpoints:
  GET  /           — Web chat UI (index.html)
  POST /chat       — Multi-turn chat (session_id 지원)
  DELETE /chat/session/{id} — 세션 초기화
  GET  /health     — liveness check

Authentication:
  If API_KEY env var is set, all requests must include header:
    X-API-Key: <value>
  Leave API_KEY unset to disable auth (dev only).

Start:
  uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.chat import ChatHandler, build_chat_handler
from src.api.models import ChatRequest, ChatResponse
from src.api.session import SessionStore
from src.chatbot_goover_context import GooverContext, build_context
from src.retrieval.retriever import get_retriever

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_handler: ChatHandler | None = None       # used by tests (no lifespan)
_session_store: SessionStore | None = None


def _goover_response_to_chat(resp, session_id: str) -> ChatResponse:
    return ChatResponse(
        answer=resp.answer,
        domain=resp.domain,
        sources=resp.sources,
        matched_keywords=resp.matched_keywords,
        session_id=session_id,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _handler, _session_store

    retriever = get_retriever(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection=os.environ.get("QDRANT_COLLECTION", "canvas_guides"),
        embedder_prefer=os.environ.get("EMBEDDING_PROVIDER", "auto"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    _handler = build_chat_handler(retriever)

    def _ctx_factory() -> GooverContext:
        return build_context(_retriever=retriever)

    _session_store = SessionStore(factory=_ctx_factory, max_sessions=200)
    yield
    _handler = None
    _session_store = None


app = FastAPI(
    title="Canvas RAG Chatbot",
    version="0.2.0",
    lifespan=lifespan,
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("API_KEY", "")


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not _API_KEY:
        return
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "Canvas RAG Chatbot API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "handler_ready": _handler is not None,
        "sessions": _session_store.size if _session_store else 0,
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    # Production path: use session store (GooverContext, multi-turn)
    if _session_store is not None:
        sid, ctx = _session_store.get_or_create(request.session_id)
        resp = ctx.chat(
            query=request.query,
            role=request.role,
            force_domain=request.force_domain,
            top_k=request.top_k,
        )
        return _goover_response_to_chat(resp, sid)

    # Test fallback: _handler mocked by tests (lifespan not run)
    if _handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat handler not initialized",
        )
    result = _handler.handle(request)
    return result


@app.delete(
    "/chat/session/{session_id}",
    dependencies=[Depends(verify_api_key)],
    status_code=204,
)
def reset_session(session_id: str) -> None:
    """대화 기록을 초기화합니다 (세션 유지, 히스토리만 삭제)."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Session store not initialized")
    found = _session_store.reset(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
