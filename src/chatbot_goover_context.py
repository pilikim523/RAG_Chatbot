"""
Canvas RAG Chatbot — session context manager (GooverContext).

GooverContext holds a single conversation session:
  - Maintains turn-by-turn message history
  - Routes each query through DomainRouter
  - Retrieves Canvas context via CanvasRetriever
  - Calls LLM with accumulated history + retrieved context
  - Returns structured GooverResponse

Usage (standalone):
    from src.chatbot_goover_context import GooverContext, build_context

    ctx = build_context()
    resp = ctx.chat("Canvas에서 과제 제출 방법이 뭐야?")
    print(resp.answer)
    for src in resp.sources:
        print(src.title, src.source_url)

Usage (inject into FastAPI or existing pipeline):
    ctx = build_context(_retriever=my_retriever, _llm_client=my_client)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from src.api.chat import (
    _SYSTEM_PROMPT,
    _NOT_CANVAS_ANSWER,
    _build_context_block,
    _is_analysis_query,
    _sources_from_results,
    _sources_from_web,
)
from src.api.models import SourceRef
from src.retrieval.retriever import CanvasRetriever, get_retriever
from src.retrieval.router import Domain, DomainRouter, RouteDecision
from src.retrieval.web_search import (
    TavilySearcher,
    build_web_context_block,
    get_web_searcher,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GooverTurn:
    """A single conversation turn."""
    role: str          # "user" or "assistant"
    content: str


@dataclass
class GooverResponse:
    answer: str
    domain: Domain
    sources: list[SourceRef]
    matched_keywords: list[str]
    turn_index: int    # 0-based index of this turn in the session


# ---------------------------------------------------------------------------
# GooverContext
# ---------------------------------------------------------------------------

class GooverContext:
    """Stateful single-session Canvas RAG context.

    Preserves conversation history across turns so the LLM can reference
    previous exchanges. Canvas retrieval is re-run on every user turn.
    """

    DEFAULT_MIN_SCORE = 0.58

    def __init__(
        self,
        retriever: CanvasRetriever,
        llm_client: Any,
        llm_model: str,
        router: DomainRouter | None = None,
        max_history_turns: int = 10,
        min_score: float = DEFAULT_MIN_SCORE,
        web_searcher: TavilySearcher | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._model = llm_model
        self._router = router or DomainRouter()
        self._max_history = max_history_turns
        self._min_score = min_score
        self._web_searcher = web_searcher
        self._history: list[GooverTurn] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        role: str | None = None,
        force_domain: Domain | None = None,
        top_k: int = 5,
    ) -> GooverResponse:
        """Process one user turn and return GooverResponse."""
        turn_index = len([t for t in self._history if t.role == "user"])
        decision: RouteDecision = self._router.route(query, force_domain=force_domain)

        # 멀티턴: 이미 Canvas 대화 이력이 있으면 팔로업 메시지도 Canvas로 유지
        if not decision.is_canvas and self._history:
            decision = RouteDecision(domain="canvas", matched_keywords=["(이전 대화 컨텍스트)"])

        if not decision.is_canvas:
            self._history.append(GooverTurn(role="user", content=query))
            self._history.append(GooverTurn(role="assistant", content=_NOT_CANVAS_ANSWER))
            return GooverResponse(
                answer=_NOT_CANVAS_ANSWER,
                domain=decision.domain,
                sources=[],
                matched_keywords=decision.matched_keywords,
                turn_index=turn_index,
            )

        product_filter = decision.product_hint
        results = self._retriever.search(query, top_k=top_k, role=role, min_score=self._min_score, product=product_filter)
        if not results and role:
            results = self._retriever.search(query, top_k=top_k, min_score=self._min_score, product=product_filter)
        if not results and product_filter:
            results = self._retriever.search(query, top_k=top_k, role=role, min_score=self._min_score)

        # RAG 신뢰도 확인: 최고 점수 < 0.60이거나 결과 없으면 웹 검색 보조
        _WEB_FALLBACK_THRESHOLD = 0.60
        from src.retrieval.web_search import WebSearchResult
        best_rag_score = max((r.score for r in results), default=0.0)
        web_results: list[WebSearchResult] = []
        if self._web_searcher and best_rag_score < _WEB_FALLBACK_THRESHOLD:
            web_results = self._web_searcher.search(query, top_k=top_k)

        context_parts = []
        if results:
            context_parts.append(_build_context_block(results))
        if web_results:
            context_parts.append(build_web_context_block(web_results))
        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant context found.)"

        messages = self._build_messages(query, context_block)
        completion = self._llm.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        answer = completion.choices[0].message.content or ""

        self._history.append(GooverTurn(role="user", content=query))
        self._history.append(GooverTurn(role="assistant", content=answer))

        sources = _sources_from_results(results) + _sources_from_web(web_results)

        return GooverResponse(
            answer=answer,
            domain=decision.domain,
            sources=sources,
            matched_keywords=decision.matched_keywords,
            turn_index=turn_index,
        )

    def stream_chat(
        self,
        query: str,
        role: str | None = None,
        force_domain: Domain | None = None,
        top_k: int = 15,
    ):
        """SSE 이벤트를 yield하는 동기 제너레이터.

        각 이벤트는 ``data: {...}\\n\\n`` 형식의 SSE 문자열이다.
        이벤트 타입:
          - status       : 진행 상태 메시지
          - source_found : 검색된 문서 1건
          - token        : LLM 스트리밍 토큰
          - done         : 최종 완료 (sources, domain 포함)
        """
        import json

        def evt(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield evt({"type": "status", "message": "질문을 분석하고 있습니다..."})
        decision: RouteDecision = self._router.route(query, force_domain=force_domain)

        # 멀티턴: 이미 Canvas 대화 이력이 있으면 팔로업 메시지도 Canvas로 유지
        if not decision.is_canvas and self._history:
            decision = RouteDecision(domain="canvas", matched_keywords=["(이전 대화 컨텍스트)"])

        if not decision.is_canvas:
            self._history.append(GooverTurn(role="user", content=query))
            self._history.append(GooverTurn(role="assistant", content=_NOT_CANVAS_ANSWER))
            yield evt({
                "type": "done",
                "answer": _NOT_CANVAS_ANSWER,
                "domain": decision.domain,
                "sources": [],
                "matched_keywords": decision.matched_keywords,
            })
            return

        yield evt({"type": "status", "message": "Canvas 공식 문서에서 관련 내용을 검색하고 있습니다..."})
        product_filter = decision.product_hint
        results = self._retriever.search(query, top_k=top_k, role=role, min_score=self._min_score, product=product_filter)
        if not results and role:
            results = self._retriever.search(query, top_k=top_k, min_score=self._min_score, product=product_filter)
        if not results and product_filter:
            results = self._retriever.search(query, top_k=top_k, role=role, min_score=self._min_score)

        for r in results:
            yield evt({
                "type": "source_found",
                "title": r.title or "Canvas Guide",
                "url": r.source_url,
                "score": round(r.score, 3),
            })

        # Web search fallback
        _WEB_FALLBACK_THRESHOLD = 0.60
        from src.retrieval.web_search import WebSearchResult
        best_rag_score = max((r.score for r in results), default=0.0)
        web_results: list[WebSearchResult] = []
        if self._web_searcher and best_rag_score < _WEB_FALLBACK_THRESHOLD:
            yield evt({"type": "status", "message": "웹에서 추가 정보를 검색하고 있습니다..."})
            web_results = self._web_searcher.search(query, top_k=top_k)

        context_parts = []
        if results:
            context_parts.append(_build_context_block(results))
        if web_results:
            context_parts.append(build_web_context_block(web_results))
        context_block = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant context found.)"

        yield evt({"type": "status", "message": f"검색된 {len(results)}개 문서를 바탕으로 답변을 생성하고 있습니다..."})

        messages = self._build_messages(query, context_block)
        stream = self._llm.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                yield evt({"type": "token", "content": token})

        self._history.append(GooverTurn(role="user", content=query))
        self._history.append(GooverTurn(role="assistant", content=full_answer))

        sources = _sources_from_results(results) + _sources_from_web(web_results)
        yield evt({
            "type": "done",
            "domain": decision.domain,
            "matched_keywords": decision.matched_keywords,
            "sources": [
                {
                    "title": s.title,
                    "source_url": s.source_url,
                    "score": s.score,
                    "source_type": s.source_type,
                }
                for s in sources
            ],
        })

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[GooverTurn]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return len([t for t in self._history if t.role == "user"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(self, query: str, context_block: str) -> list[dict]:
        """Build the message list for the LLM: system + history + current user turn."""
        msgs: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Include recent history (trim to max_history_turns pairs)
        history = self._history[-(self._max_history * 2):]
        for turn in history:
            msgs.append({"role": turn.role, "content": turn.content})

        # Current user turn with freshly retrieved context prepended
        if _is_analysis_query(query):
            instruction = (
                "아래는 Canvas 생태계 구현 가능성 분석 요청입니다.\n"
                "반드시 CASE A 형식으로 답변하세요:\n"
                "- SFR 섹션별로 ## 헤딩 + | 항목 | 판정 | 비고 | 테이블 출력\n"
                "- 판정: ✅ 지원 / ⚠️(a) 복합 API / ⚠️(b) LTI 연동 / ⚠️(c) 커스터마이징 / ❌ 미지원 / 🔍 확인필요\n"
                "- 비고: [기능 설명] + [API 엔드포인트] + [근거번호] 세 부분 반드시 포함\n"
                "- 복합 API 예시: ①POST /api/v1/courses/:id/assignments → ②GET /api/v1/courses/:id/sections [N]\n"
                "- LTI 연동 예시: Canvas LMS 자체 기능 아님. Panopto LTI 연동으로 이어보기 지원 [N]\n"
                "- Panopto/Turnitin 등 LTI 도구 기능은 절대 Canvas LMS 네이티브 기능으로 표시 금지\n"
                "- 마지막에 ## 요약 테이블(⚠️(b) LTI 연동 필요 행 포함) + ### 핵심 gap 목록\n\n"
            )
            query = instruction + query
        user_content = f"Context:\n\n{context_block}\n\nQuestion: {query}"
        msgs.append({"role": "user", "content": user_content})
        return msgs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_context(
    qdrant_url: str | None = None,
    collection: str | None = None,
    max_history_turns: int = 10,
    _retriever: CanvasRetriever | None = None,
    _llm_client: Any = None,
) -> GooverContext:
    """Build a GooverContext from environment variables.

    Injection parameters (_retriever, _llm_client) are provided for tests.
    """
    retriever = _retriever or get_retriever(
        qdrant_url=qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection=collection or os.environ.get("QDRANT_COLLECTION", "canvas_guides"),
        embedder_prefer=os.environ.get("EMBEDDING_PROVIDER", "auto"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    if _llm_client is not None:
        llm_client = _llm_client
        model = os.environ.get("OLLAMA_MODEL", "exaone3.5:7.8b")
    elif os.environ.get("LLM_PROVIDER", "ollama").lower() == "ollama":
        from openai import OpenAI
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
        model = os.environ.get("OLLAMA_MODEL", "exaone3.5:7.8b")
        llm_client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        from openai import OpenAI
        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    return GooverContext(
        retriever=retriever,
        llm_client=llm_client,
        llm_model=model,
        max_history_turns=max_history_turns,
        web_searcher=get_web_searcher(),
    )
