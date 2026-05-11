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

# .env 파일 자동 로드 (개발 환경)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.chat import ChatHandler, build_chat_handler
from src.api.models import (
    ChatRequest,
    ChatResponse,
    SessionListItem,
    SessionHistoryResponse,
    TurnItem,
)
from src.api.session import SessionStore
from src.api import persistence
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

    try:
        zoom_retriever = get_retriever(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            collection="zoom_docs",
            embedder_prefer=os.environ.get("EMBEDDING_PROVIDER", "auto"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    except Exception:
        zoom_retriever = None

    def _ctx_factory() -> GooverContext:
        return build_context(_retriever=retriever, _zoom_retriever=zoom_retriever)

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
        return FileResponse(
            str(html_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
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
        _session_store.persist(sid)
        return _goover_response_to_chat(resp, sid)

    # Test fallback: _handler mocked by tests (lifespan not run)
    if _handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat handler not initialized",
        )
    result = _handler.handle(request)
    return result


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE 스트리밍 채팅. 진행 상태·토큰·출처를 실시간으로 전달한다."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Session store not initialized")
    sid, ctx = _session_store.get_or_create(request.session_id)

    def generate():
        import json
        for event_str in ctx.stream_chat(
            query=request.query,
            role=request.role,
            force_domain=request.force_domain,
            top_k=request.top_k,
        ):
            # done 이벤트에 session_id 주입 후 히스토리 저장
            if '"type": "done"' in event_str:
                raw = event_str.replace("data: ", "", 1).strip()
                data = json.loads(raw)
                data["session_id"] = sid
                _session_store.persist(sid)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            else:
                yield event_str

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.delete(
    "/chat/session/{session_id}",
    dependencies=[Depends(verify_api_key)],
    status_code=204,
)
def delete_session(session_id: str) -> None:
    """세션을 완전히 삭제합니다 (히스토리 포함)."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Session store not initialized")
    _session_store.delete(session_id)


@app.post(
    "/chat/session/{session_id}/reset",
    dependencies=[Depends(verify_api_key)],
    status_code=204,
)
def reset_session(session_id: str) -> None:
    """히스토리만 초기화합니다 (세션 ID 유지)."""
    if _session_store is None:
        raise HTTPException(status_code=503, detail="Session store not initialized")
    found = _session_store.reset(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get(
    "/chat/sessions",
    dependencies=[Depends(verify_api_key)],
    response_model=list[SessionListItem],
)
def list_sessions() -> list[SessionListItem]:
    """최근 대화 세션 목록을 반환합니다."""
    try:
        rows = persistence.list_sessions(limit=100)
        return [SessionListItem(**r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


@app.get(
    "/chat/session/{session_id}/history",
    dependencies=[Depends(verify_api_key)],
    response_model=SessionHistoryResponse,
)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    """특정 세션의 전체 대화 히스토리를 반환합니다."""
    try:
        data = persistence.get_session_history(session_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionHistoryResponse(
        id=data["id"],
        title=data["title"],
        turns=[TurnItem(role=t["role"], content=t["content"]) for t in data["turns"]],
    )
