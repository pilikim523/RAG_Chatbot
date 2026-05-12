"""Tests for src/retrieval/retriever.py — no real Qdrant or GPU required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.indexing.embedder import BaseEmbedder
from src.retrieval.retriever import (
    CanvasRetriever,
    SearchResult,
    _build_filter,
    get_retriever,
)

BASE_URL = "https://community.instructure.com/en/kb/articles/661210-submit"


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------

class FakeEmbedder(BaseEmbedder):
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_hit(
    source_url: str = BASE_URL,
    title: str = "Submit Assignment",
    role: str = "student",
    category: str = "canvas-lms-student-guide",
    text: str = "Click Assignments in the navigation.",
    score: float = 0.92,
    chunk_index: int = 0,
    chunk_total: int = 3,
) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "source_url": source_url,
        "canonical_url": source_url,
        "title": title,
        "product": "canvas",
        "guide": "661210-submit",
        "category": category,
        "role": role,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "content_hash": "abc123",
    }
    return hit


def _make_client(hits: list | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.points = hits or []
    client.query_points.return_value = response
    return client


def _make_retriever(hits: list | None = None) -> CanvasRetriever:
    return CanvasRetriever(
        embedder=FakeEmbedder(),
        client=_make_client(hits),
        collection="canvas_guides",
    )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_from_qdrant_point(self):
        hit = _make_hit()
        result = SearchResult.from_qdrant_point(hit)
        assert result.source_url == BASE_URL
        assert result.title == "Submit Assignment"
        assert result.role == "student"
        assert result.score == 0.92

    def test_missing_payload_fields_use_defaults(self):
        hit = MagicMock()
        hit.score = 0.5
        hit.payload = {}
        result = SearchResult.from_qdrant_point(hit)
        assert result.source_url == ""
        assert result.product == "canvas"
        assert result.text == ""

    def test_score_is_float(self):
        hit = _make_hit(score=0.88)
        result = SearchResult.from_qdrant_point(hit)
        assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------

class TestBuildFilter:
    def test_none_when_no_constraints(self):
        assert _build_filter() is None

    def test_role_filter(self):
        f = _build_filter(role="student")
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "role"

    def test_multiple_filters(self):
        f = _build_filter(role="instructor", category="canvas-lms-instructor-guide")
        assert f is not None
        assert len(f.must) == 2

    def test_product_filter(self):
        f = _build_filter(product="canvas")
        assert f is not None
        assert f.must[0].key == "product"


# ---------------------------------------------------------------------------
# CanvasRetriever.search
# ---------------------------------------------------------------------------

class TestCanvasRetrieverSearch:
    def test_returns_search_results(self):
        # source_url이 다른 2개 hit → dedup 후 2개 반환
        hits = [
            _make_hit(source_url=BASE_URL + "/a", score=0.9),
            _make_hit(source_url=BASE_URL + "/b", score=0.8),
        ]
        retriever = _make_retriever(hits)
        results = retriever.search("assignment submission")
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_empty_query_returns_empty(self):
        retriever = _make_retriever()
        results = retriever.search("")
        assert results == []

    def test_whitespace_only_query_returns_empty(self):
        retriever = _make_retriever()
        results = retriever.search("   ")
        assert results == []

    def test_no_results_from_qdrant(self):
        retriever = _make_retriever(hits=[])
        results = retriever.search("some query")
        assert results == []

    def test_results_ordered_by_score(self):
        # source_url이 다른 3개 → dedup 후 score 내림차순
        hits = [
            _make_hit(source_url=BASE_URL + "/x", score=0.9),
            _make_hit(source_url=BASE_URL + "/y", score=0.7),
            _make_hit(source_url=BASE_URL + "/z", score=0.85),
        ]
        retriever = _make_retriever(hits)
        results = retriever.search("query")
        assert results[0].score == 0.9
        assert results[1].score == 0.85

    def test_passes_top_k_to_client(self):
        # retriever는 dedup을 위해 fetch_k = min(top_k * 3, 60) 으로 요청
        client = _make_client()
        retriever = CanvasRetriever(FakeEmbedder(), client, "canvas_guides")
        retriever.search("query", top_k=3)
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs["limit"] == min(3 * 3, 60)  # fetch_k = 9

    def test_passes_role_filter_when_set(self):
        client = _make_client()
        retriever = CanvasRetriever(FakeEmbedder(), client, "canvas_guides")
        retriever.search("query", role="student")
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None

    def test_no_filter_when_no_constraints(self):
        client = _make_client()
        retriever = CanvasRetriever(FakeEmbedder(), client, "canvas_guides")
        retriever.search("query")
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is None

    def test_embed_called_with_query(self):
        embedder = FakeEmbedder()
        original_embed_one = embedder.embed_one

        called_with = []
        def tracking_embed_one(text):
            called_with.append(text)
            return original_embed_one(text)

        embedder.embed_one = tracking_embed_one
        retriever = CanvasRetriever(embedder, _make_client(), "canvas_guides")
        retriever.search("my test query")
        assert called_with == ["my test query"]

    def test_result_text_matches_payload(self):
        hits = [_make_hit(text="Submit by clicking the button.")]
        retriever = _make_retriever(hits)
        results = retriever.search("submit assignment")
        assert results[0].text == "Submit by clicking the button."


# ---------------------------------------------------------------------------
# CanvasRetriever.search_multi_role
# ---------------------------------------------------------------------------

class TestSearchMultiRole:
    def test_returns_results(self):
        hits = [_make_hit(role="student"), _make_hit(role="instructor")]
        client = _make_client(hits)
        retriever = CanvasRetriever(FakeEmbedder(), client, "canvas_guides")
        results = retriever.search_multi_role("assignment", roles=["student", "instructor"])
        assert len(results) == 2

    def test_empty_query_returns_empty(self):
        retriever = _make_retriever()
        results = retriever.search_multi_role("", roles=["student"])
        assert results == []

    def test_passes_role_filter_with_match_any(self):
        client = _make_client()
        retriever = CanvasRetriever(FakeEmbedder(), client, "canvas_guides")
        retriever.search_multi_role("query", roles=["student", "instructor"])
        call_kwargs = client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None


# ---------------------------------------------------------------------------
# get_retriever factory
# ---------------------------------------------------------------------------

class TestGetRetriever:
    def test_returns_canvas_retriever(self):
        embedder = FakeEmbedder()
        client = MagicMock()
        retriever = get_retriever(_embedder=embedder, _client=client)
        assert isinstance(retriever, CanvasRetriever)

    def test_uses_injected_embedder(self):
        embedder = FakeEmbedder()
        client = MagicMock()
        retriever = get_retriever(_embedder=embedder, _client=client)
        assert retriever._embedder is embedder

    def test_uses_injected_client(self):
        client = MagicMock()
        retriever = get_retriever(_embedder=FakeEmbedder(), _client=client)
        assert retriever._client is client

    def test_custom_collection(self):
        retriever = get_retriever(
            collection="custom_col",
            _embedder=FakeEmbedder(),
            _client=MagicMock(),
        )
        assert retriever._collection == "custom_col"
