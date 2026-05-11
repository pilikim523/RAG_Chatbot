"""
Stage 5: Qdrant vector retriever for Canvas guides.

Usage:
    from src.retrieval.retriever import CanvasRetriever, get_retriever

    retriever = get_retriever()
    results = retriever.search("Canvas assignment due date", top_k=5)
    for r in results:
        print(r.score, r.title, r.source_url)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from src.indexing.embedder import BaseEmbedder, get_embedder

COLLECTION_NAME = "canvas_guides"
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    chunk_id: str
    source_url: str
    canonical_url: str | None
    title: str | None
    product: str
    guide: str | None
    category: str | None
    role: str | None
    chunk_index: int
    chunk_total: int
    text: str
    score: float
    word_count: int = 0
    char_count: int = 0

    @classmethod
    def from_qdrant_point(cls, point: Any) -> "SearchResult":
        p = point.payload or {}
        return cls(
            chunk_id=p.get("chunk_id", ""),
            source_url=p.get("source_url", ""),
            canonical_url=p.get("canonical_url"),
            title=p.get("title"),
            product=p.get("product", "canvas"),
            guide=p.get("guide"),
            category=p.get("category"),
            role=p.get("role"),
            chunk_index=p.get("chunk_index", 0),
            chunk_total=p.get("chunk_total", 1),
            text=p.get("text", ""),
            score=float(point.score),
            word_count=p.get("word_count", 0),
            char_count=p.get("char_count", 0),
        )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CanvasRetriever:
    """Vector search over the canvas_guides Qdrant collection."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        client: QdrantClient,
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._embedder = embedder
        self._client = client
        self._collection = collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        role: str | None = None,
        category: str | None = None,
        product: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Embed query and return top_k nearest chunks.

        Optionally filter by role, category, or product payload fields.
        min_score: drop results with score below this threshold (0.0–1.0).
        """
        if not query.strip():
            return []

        query_vector = self._embedder.embed_one(query)
        qdrant_filter = _build_filter(role=role, category=category, product=product)

        # 중복 문서 제거를 고려해 실제보다 더 많이 가져온 후 dedup
        fetch_k = min(top_k * 3, 60)
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=fetch_k,
            with_payload=True,
        )
        results = [SearchResult.from_qdrant_point(h) for h in response.points]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # source_url 기준 dedup: 같은 문서에서 최고 점수 청크만 유지
        seen_urls: dict[str, SearchResult] = {}
        for r in results:
            key = r.source_url
            if key not in seen_urls or r.score > seen_urls[key].score:
                seen_urls[key] = r
        deduped = sorted(seen_urls.values(), key=lambda r: r.score, reverse=True)
        return deduped[:top_k]

    def search_multi_role(
        self,
        query: str,
        roles: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Search across multiple roles (OR filter), deduplicated by score."""
        if not query.strip():
            return []

        query_vector = self._embedder.embed_one(query)
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="role",
                    match=MatchAny(any=roles),
                )
            ]
        )
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [SearchResult.from_qdrant_point(h) for h in response.points]


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def _build_filter(
    role: str | None = None,
    category: str | None = None,
    product: str | None = None,
) -> Filter | None:
    conditions = []
    if role:
        conditions.append(FieldCondition(key="role", match=MatchValue(value=role)))
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if product:
        conditions.append(FieldCondition(key="product", match=MatchValue(value=product)))
    return Filter(must=conditions) if conditions else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_retriever(
    qdrant_url: str = "http://localhost:6333",
    collection: str = COLLECTION_NAME,
    embedder_prefer: str = "auto",
    openai_api_key: str | None = None,
    _embedder: BaseEmbedder | None = None,
    _client: QdrantClient | None = None,
) -> CanvasRetriever:
    """Build a CanvasRetriever with the best available embedder."""
    embedder = _embedder or get_embedder(prefer=embedder_prefer, openai_api_key=openai_api_key)
    client = _client or QdrantClient(url=qdrant_url)
    return CanvasRetriever(embedder=embedder, client=client, collection=collection)
