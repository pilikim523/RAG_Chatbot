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

    DEFAULT_MIN_SCORE = 0.50

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

        results = self._retriever.search(query, top_k=top_k, role=role, min_score=self._min_score)
        if not results and role:
            results = self._retriever.search(query, top_k=top_k, min_score=self._min_score)

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
            temperature=0.2,
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
