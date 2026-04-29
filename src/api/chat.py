"""
Chat pipeline: router → retriever → (web search fallback) → LLM → Korean answer.

LLM backend: Ollama (OpenAI-compatible) or OpenAI, configured via env vars.
English source docs are passed as context; the LLM answers in Korean.
"""
from __future__ import annotations

import os
from typing import Any

from src.api.models import ChatRequest, ChatResponse, SourceRef
from src.retrieval.retriever import CanvasRetriever, SearchResult
from src.retrieval.router import DomainRouter, RouteDecision
from src.retrieval.web_search import (
    TavilySearcher,
    WebSearchResult,
    build_web_context_block,
    get_web_searcher,
)

_SYSTEM_PROMPT = """\
You are a Canvas LMS support assistant for internal use at a Korean institution.
You are given excerpts from official Canvas documentation (written in English).

LANGUAGE RULE (HIGHEST PRIORITY):
- You MUST respond ONLY in Korean (한국어).
- NEVER output Chinese, Japanese, or any language other than Korean and English.
- Canvas UI terms, button labels, and feature names stay in English with Korean alongside \
(e.g. 과제 제출(Submit Assignment), 성적부(Gradebook)).
- If you notice yourself writing Chinese characters (中文), STOP immediately and rewrite in Korean.

CONTENT RULES:
- Base your answer ONLY on the provided context. Do not use general knowledge not present in the context.
- If the context does not contain enough information, respond with exactly: \
"현재 수집된 Canvas 공식 문서에서 확인된 근거가 없습니다."
- When answering how-to questions, provide numbered step-by-step instructions.
- End every Canvas answer with a "참고 문서" section listing source titles and URLs.\
"""

_NOT_CANVAS_ANSWER = (
    "이 챗봇은 Canvas LMS 관련 질문만 답변합니다. "
    "Canvas 기능, 과제, 성적, 강좌 운영 등에 대해 질문해 주세요."
)


def _build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM prompt."""
    if not results:
        return "(No relevant context found.)"
    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r.title or 'Canvas Guide'} — {r.source_url}"
        parts.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(parts)


def _build_user_message(query: str, context: str) -> str:
    return f"Context:\n\n{context}\n\nQuestion: {query}"


def _sources_from_results(results: list[SearchResult]) -> list[SourceRef]:
    seen: set[str] = set()
    sources: list[SourceRef] = []
    for r in results:
        if r.source_url not in seen:
            seen.add(r.source_url)
            sources.append(SourceRef(
                title=r.title,
                source_url=r.source_url,
                score=round(r.score, 4),
                source_type="rag",
            ))
    return sources


def _sources_from_web(results: list[WebSearchResult]) -> list[SourceRef]:
    seen: set[str] = set()
    sources: list[SourceRef] = []
    for r in results:
        if r.source_url not in seen:
            seen.add(r.source_url)
            sources.append(SourceRef(
                title=r.title,
                source_url=r.source_url,
                score=round(r.score, 4),
                source_type="web",
            ))
    return sources


class ChatHandler:
    """Orchestrates router → retriever → (web fallback) → LLM for a single chat turn."""

    # Cosine similarity threshold: results below this score are considered
    # off-topic. Kept at 0.50 to accommodate Korean→English cross-lingual
    # queries (bge-m3 scores ~0.53 for correct matches vs ~0.67 for English).
    DEFAULT_MIN_SCORE = 0.50

    def __init__(
        self,
        retriever: CanvasRetriever,
        llm_client: Any,
        llm_model: str,
        router: DomainRouter | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        web_searcher: TavilySearcher | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._model = llm_model
        self._router = router or DomainRouter()
        self._min_score = min_score
        self._web_searcher = web_searcher

    def handle(self, request: ChatRequest) -> ChatResponse:
        decision: RouteDecision = self._router.route(
            request.query, force_domain=request.force_domain
        )

        if not decision.is_canvas:
            return ChatResponse(
                answer=_NOT_CANVAS_ANSWER,
                domain=decision.domain,
                sources=[],
                matched_keywords=decision.matched_keywords,
            )

        # 1. RAG 검색
        results = self._retriever.search(
            request.query,
            top_k=request.top_k,
            role=request.role,
            min_score=self._min_score,
        )
        if not results and request.role:
            results = self._retriever.search(
                request.query, top_k=request.top_k, min_score=self._min_score
            )

        # 2. RAG 신뢰도 확인: 최고 점수 < 0.60이거나 결과 없으면 웹 검색 보조
        _WEB_FALLBACK_THRESHOLD = 0.60
        best_rag_score = max((r.score for r in results), default=0.0)
        web_results: list[WebSearchResult] = []
        if self._web_searcher and best_rag_score < _WEB_FALLBACK_THRESHOLD:
            web_results = self._web_searcher.search(request.query, top_k=request.top_k)

        # 3. 컨텍스트 조합 (RAG + 웹 병합, 또는 웹 단독)
        context_parts = []
        if results:
            context_parts.append(_build_context_block(results))
        if web_results:
            context_parts.append(build_web_context_block(web_results))
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant context found.)"

        user_msg = _build_user_message(request.query, context)
        completion = self._llm.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content or ""

        sources = _sources_from_results(results) + _sources_from_web(web_results)

        return ChatResponse(
            answer=answer,
            domain=decision.domain,
            sources=sources,
            matched_keywords=decision.matched_keywords,
        )


# ---------------------------------------------------------------------------
# Factory — reads env vars
# ---------------------------------------------------------------------------

def build_chat_handler(
    retriever: CanvasRetriever,
    _llm_client: Any = None,
) -> ChatHandler:
    """Build ChatHandler from environment variables."""
    if _llm_client is not None:
        model = os.environ.get("OLLAMA_MODEL", "exaone3.5:7.8b")
        return ChatHandler(retriever=retriever, llm_client=_llm_client, llm_model=model)

    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from openai import OpenAI
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
        model = os.environ.get("OLLAMA_MODEL", "exaone3.5:7.8b")
        client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        from openai import OpenAI
        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    web_searcher = get_web_searcher()

    return ChatHandler(
        retriever=retriever,
        llm_client=client,
        llm_model=model,
        web_searcher=web_searcher,
    )
