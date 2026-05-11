"""
Web search: SearXNG (self-hosted, preferred) or Tavily API (fallback).

Canvas RAG 점수가 낮을 때 혹은 일반 웹 질문에 대해 웹 검색 결과를 LLM 컨텍스트로 제공한다.
SEARXNG_URL 환경변수가 설정되면 SearXNG를 우선 사용한다.
TAVILY_API_KEY만 있으면 Tavily를 사용한다. 둘 다 없으면 비활성화된다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# Instructure 공식 도메인 우선 검색 (Tavily용)
_INSTRUCTURE_DOMAINS = [
    "community.instructure.com",
    "developerdocs.instructure.com",
    "instructure.com",
    "canvas.instructure.com",
    "help.instructure.com",
]


@dataclass
class WebSearchResult:
    title: str | None
    source_url: str
    content: str      # 검색 결과 본문 (LLM 컨텍스트용)
    score: float      # relevance score (0~1)


class TavilySearcher:
    """Tavily 기반 웹 검색 클라이언트."""

    def __init__(self, api_key: str, restrict_to_instructure: bool = True) -> None:
        from tavily import TavilyClient
        self._client = TavilyClient(api_key=api_key)
        self._restrict = restrict_to_instructure

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        base_kwargs: dict = {
            "query": query,
            "max_results": top_k,
            "include_answer": False,
            "include_raw_content": False,
        }

        def _do_search(extra: dict) -> list[WebSearchResult]:
            try:
                resp = self._client.search(**{**base_kwargs, **extra})
            except Exception:
                return []
            return [
                WebSearchResult(
                    title=r.get("title"),
                    source_url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=round(float(r.get("score", 0.0)), 4),
                )
                for r in resp.get("results", [])
            ]

        if self._restrict:
            # 1차: Instructure 도메인 한정 (한국어 쿼리도 지원)
            results = _do_search({"include_domains": _INSTRUCTURE_DOMAINS})
            # 도메인 한정으로 결과 없으면 전체 웹으로 재시도
            if not results:
                results = _do_search({})
            return results

        return _do_search({})


class SearXNGSearcher:
    """SearXNG 기반 웹 검색 클라이언트 (자체 호스팅)."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        import httpx
        try:
            resp = httpx.get(
                f"{self._base_url}/search",
                params={"q": query, "format": "json", "language": "auto"},
                timeout=10.0,
                headers={"User-Agent": "canvas-rag-chatbot/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log.warning("SearXNG search failed: %s", e)
            return []

        results = []
        for i, r in enumerate(data.get("results", [])[:top_k]):
            raw_score = r.get("score")
            score = float(raw_score) if raw_score is not None else round(1.0 - i * 0.1, 2)
            results.append(WebSearchResult(
                title=r.get("title"),
                source_url=r.get("url", ""),
                content=r.get("content", ""),
                score=round(max(0.0, min(1.0, score)), 4),
            ))
        return results


def get_web_searcher(
    api_key: str | None = None,
    restrict_to_instructure: bool = True,
) -> SearXNGSearcher | TavilySearcher | None:
    """사용 가능한 웹 검색 클라이언트를 반환. SearXNG > Tavily > None 순으로 선택."""
    searxng_url = os.environ.get("SEARXNG_URL")
    if searxng_url:
        _log.debug("Using SearXNG at %s", searxng_url)
        return SearXNGSearcher(base_url=searxng_url)

    key = api_key or os.environ.get("TAVILY_API_KEY")
    if key:
        _log.debug("Using Tavily search")
        return TavilySearcher(api_key=key, restrict_to_instructure=restrict_to_instructure)

    return None


def build_web_context_block(results: list[WebSearchResult]) -> str:
    """웹 검색 결과를 LLM 컨텍스트 블록으로 포맷."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        header = f"[Web {i}] {r.title or '웹 검색 결과'} — {r.source_url}"
        parts.append(f"{header}\n{r.content}")
    return "\n\n---\n\n".join(parts)
