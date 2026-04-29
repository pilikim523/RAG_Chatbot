"""
Web search fallback using Tavily API.

RAG 검색 결과가 없거나 부족할 때 Instructure 공식 도메인을 우선으로 웹 검색한다.
TAVILY_API_KEY 환경변수가 없으면 조용히 비활성화된다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# Instructure 공식 도메인 우선 검색
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
    score: float      # Tavily relevance score (0~1)


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


def get_web_searcher(
    api_key: str | None = None,
    restrict_to_instructure: bool = True,
) -> TavilySearcher | None:
    """환경변수 또는 인자에서 API 키를 읽어 TavilySearcher를 반환. 키 없으면 None."""
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    return TavilySearcher(api_key=key, restrict_to_instructure=restrict_to_instructure)


def build_web_context_block(results: list[WebSearchResult]) -> str:
    """웹 검색 결과를 LLM 컨텍스트 블록으로 포맷."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        header = f"[Web {i}] {r.title or '웹 검색 결과'} — {r.source_url}"
        parts.append(f"{header}\n{r.content}")
    return "\n\n---\n\n".join(parts)
