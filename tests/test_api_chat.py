"""Tests for src/api/chat.py — no real Qdrant, Ollama, or GPU required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.chat import (
    ChatHandler,
    _NOT_CANVAS_ANSWER,
    _build_context_block,
    _build_user_message,
    _sources_from_results,
)
from src.api.models import ChatRequest, ChatResponse
from src.indexing.embedder import BaseEmbedder
from src.retrieval.retriever import CanvasRetriever, SearchResult
from src.retrieval.router import DomainRouter

BASE_URL = "https://community.instructure.com/en/kb/articles/661210-submit"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbedder(BaseEmbedder):
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_result(
    source_url: str = BASE_URL,
    title: str = "Submit Assignment",
    role: str = "student",
    text: str = "Click Assignments. Then click Submit Assignment.",
    score: float = 0.91,
) -> SearchResult:
    return SearchResult(
        chunk_id="abc123",
        source_url=source_url,
        canonical_url=source_url,
        title=title,
        product="canvas",
        guide="661210-submit",
        category="canvas-lms-student-guide",
        role=role,
        chunk_index=0,
        chunk_total=1,
        text=text,
        score=score,
    )


def _make_retriever(results: list[SearchResult] | None = None) -> CanvasRetriever:
    retriever = MagicMock(spec=CanvasRetriever)
    retriever.search.return_value = results or []
    retriever._client = MagicMock()
    return retriever


def _make_llm_client(answer: str = "테스트 답변입니다.") -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = answer
    client.chat.completions.create.return_value.choices = [choice]
    return client


def _make_handler(
    results: list[SearchResult] | None = None,
    answer: str = "테스트 답변입니다.",
) -> ChatHandler:
    return ChatHandler(
        retriever=_make_retriever(results),
        llm_client=_make_llm_client(answer),
        llm_model="qwen2.5:7b-instruct",
    )


# ---------------------------------------------------------------------------
# _build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock:
    def test_empty_results(self):
        assert "(No relevant context found.)" in _build_context_block([])

    def test_includes_title_and_url(self):
        r = _make_result(title="Submit Assignment", source_url=BASE_URL)
        block = _build_context_block([r])
        assert "Submit Assignment" in block
        assert BASE_URL in block

    def test_includes_chunk_text(self):
        r = _make_result(text="Click the submit button.")
        block = _build_context_block([r])
        assert "Click the submit button." in block

    def test_multiple_results_separated(self):
        results = [_make_result(text=f"Text {i}") for i in range(3)]
        block = _build_context_block(results)
        assert "[1]" in block
        assert "[2]" in block
        assert "[3]" in block

    def test_numbered_sequentially(self):
        results = [_make_result(), _make_result()]
        block = _build_context_block(results)
        assert block.index("[1]") < block.index("[2]")


# ---------------------------------------------------------------------------
# _sources_from_results
# ---------------------------------------------------------------------------

class TestSourcesFromResults:
    def test_deduplicates_by_url(self):
        results = [_make_result(), _make_result()]  # same URL
        sources = _sources_from_results(results)
        assert len(sources) == 1

    def test_different_urls_kept(self):
        r1 = _make_result(source_url=BASE_URL + "/1")
        r2 = _make_result(source_url=BASE_URL + "/2")
        sources = _sources_from_results([r1, r2])
        assert len(sources) == 2

    def test_score_rounded(self):
        r = _make_result(score=0.912345)
        sources = _sources_from_results([r])
        assert sources[0].score == round(0.912345, 4)

    def test_title_preserved(self):
        r = _make_result(title="My Guide")
        sources = _sources_from_results([r])
        assert sources[0].title == "My Guide"

    def test_empty_results(self):
        assert _sources_from_results([]) == []


# ---------------------------------------------------------------------------
# ChatHandler.handle — Canvas domain
# ---------------------------------------------------------------------------

class TestChatHandlerCanvas:
    def test_returns_chat_response(self):
        handler = _make_handler(results=[_make_result()])
        resp = handler.handle(ChatRequest(query="How do I submit an assignment?"))
        assert isinstance(resp, ChatResponse)

    def test_domain_is_canvas(self):
        handler = _make_handler(results=[_make_result()])
        resp = handler.handle(ChatRequest(query="assignment submission"))
        assert resp.domain == "canvas"

    def test_answer_from_llm(self):
        handler = _make_handler(results=[_make_result()], answer="과제 제출 방법입니다.")
        resp = handler.handle(ChatRequest(query="assignment submission"))
        assert resp.answer == "과제 제출 방법입니다."

    def test_sources_populated(self):
        handler = _make_handler(results=[_make_result()])
        resp = handler.handle(ChatRequest(query="submit assignment"))
        assert len(resp.sources) == 1
        assert resp.sources[0].source_url == BASE_URL

    def test_matched_keywords_present(self):
        handler = _make_handler()
        resp = handler.handle(ChatRequest(query="assignment due date"))
        assert "assignment" in resp.matched_keywords

    def test_llm_called_with_system_prompt(self):
        llm = _make_llm_client()
        handler = ChatHandler(
            retriever=_make_retriever([_make_result()]),
            llm_client=llm,
            llm_model="test-model",
        )
        handler.handle(ChatRequest(query="quiz creation"))
        call_messages = llm.chat.completions.create.call_args[1]["messages"]
        assert call_messages[0]["role"] == "system"
        assert "Korean" in call_messages[0]["content"]

    def test_role_filter_passed_to_retriever(self):
        retriever = _make_retriever([_make_result()])  # non-empty so fallback isn't triggered
        llm = _make_llm_client()
        handler = ChatHandler(retriever=retriever, llm_client=llm, llm_model="m")
        handler.handle(ChatRequest(query="assignment", role="student"))
        first_call_kwargs = retriever.search.call_args_list[0][1]
        assert first_call_kwargs.get("role") == "student"

    def test_top_k_passed_to_retriever(self):
        retriever = _make_retriever()
        llm = _make_llm_client()
        handler = ChatHandler(retriever=retriever, llm_client=llm, llm_model="m")
        handler.handle(ChatRequest(query="assignment", top_k=3))
        call_kwargs = retriever.search.call_args[1]
        assert call_kwargs["top_k"] == 3

    def test_no_results_still_calls_llm(self):
        llm = _make_llm_client()
        handler = ChatHandler(
            retriever=_make_retriever([]),
            llm_client=llm,
            llm_model="m",
        )
        handler.handle(ChatRequest(query="canvas quiz"))
        assert llm.chat.completions.create.called


# ---------------------------------------------------------------------------
# ChatHandler.handle — non-Canvas domain
# ---------------------------------------------------------------------------

class TestChatHandlerNonCanvas:
    def test_general_query_no_llm_call(self):
        llm = _make_llm_client()
        handler = ChatHandler(
            retriever=_make_retriever(),
            llm_client=llm,
            llm_model="m",
        )
        handler.handle(ChatRequest(query="오늘 날씨 어때요?"))
        assert not llm.chat.completions.create.called

    def test_general_query_returns_redirect_answer(self):
        handler = _make_handler()
        resp = handler.handle(ChatRequest(query="오늘 날씨 어때요?"))
        assert resp.answer == _NOT_CANVAS_ANSWER

    def test_general_domain_in_response(self):
        handler = _make_handler()
        resp = handler.handle(ChatRequest(query="오늘 날씨 어때요?"))
        assert resp.domain == "web"

    def test_force_canvas_calls_llm(self):
        llm = _make_llm_client()
        handler = ChatHandler(
            retriever=_make_retriever([_make_result()]),
            llm_client=llm,
            llm_model="m",
        )
        handler.handle(ChatRequest(query="오늘 날씨?", force_domain="canvas"))
        assert llm.chat.completions.create.called
