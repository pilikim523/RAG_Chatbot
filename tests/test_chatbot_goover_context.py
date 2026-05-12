"""Tests for src/chatbot_goover_context.py"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.chat import _NOT_CANVAS_ANSWER
from src.api.models import SourceRef
from src.chatbot_goover_context import GooverContext, GooverResponse, GooverTurn, build_context
from src.indexing.embedder import BaseEmbedder
from src.retrieval.retriever import CanvasRetriever, SearchResult

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


def _make_search_result(text: str = "Click Submit Assignment.") -> SearchResult:
    return SearchResult(
        chunk_id="abc",
        source_url=BASE_URL,
        canonical_url=BASE_URL,
        title="Submit Assignment",
        product="canvas",
        guide="661210-submit",
        category="canvas-lms-student-guide",
        role="student",
        chunk_index=0,
        chunk_total=1,
        text=text,
        score=0.92,
    )


def _make_retriever(results: list | None = None) -> CanvasRetriever:
    r = MagicMock(spec=CanvasRetriever)
    r.search.return_value = results or []
    return r


def _make_llm(answer: str = "Canvas 답변입니다.") -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = answer
    client.chat.completions.create.return_value.choices = [choice]
    return client


def _make_ctx(
    results: list | None = None,
    answer: str = "Canvas 답변입니다.",
    max_history_turns: int = 10,
) -> GooverContext:
    return GooverContext(
        retriever=_make_retriever(results),
        llm_client=_make_llm(answer),
        llm_model="qwen2.5:7b-instruct",
        max_history_turns=max_history_turns,
    )


# ---------------------------------------------------------------------------
# GooverTurn / GooverResponse basics
# ---------------------------------------------------------------------------

class TestGooverTurn:
    def test_role_and_content(self):
        t = GooverTurn(role="user", content="hello")
        assert t.role == "user"
        assert t.content == "hello"


class TestGooverResponse:
    def test_fields(self):
        r = GooverResponse(
            answer="답변", domain="canvas",
            sources=[], matched_keywords=["assignment"], turn_index=0,
        )
        assert r.answer == "답변"
        assert r.domain == "canvas"
        assert r.turn_index == 0


# ---------------------------------------------------------------------------
# GooverContext.chat — Canvas domain
# ---------------------------------------------------------------------------

class TestGooverContextCanvas:
    def test_returns_goover_response(self):
        ctx = _make_ctx(results=[_make_search_result()])
        resp = ctx.chat("How do I submit an assignment?")
        assert isinstance(resp, GooverResponse)

    def test_domain_is_canvas(self):
        ctx = _make_ctx(results=[_make_search_result()])
        resp = ctx.chat("assignment submission")
        assert resp.domain == "canvas"

    def test_answer_from_llm(self):
        ctx = _make_ctx(results=[_make_search_result()], answer="과제 제출 방법입니다.")
        resp = ctx.chat("assignment")
        assert resp.answer == "과제 제출 방법입니다."

    def test_sources_populated(self):
        ctx = _make_ctx(results=[_make_search_result()])
        resp = ctx.chat("assignment")
        assert len(resp.sources) >= 1
        assert resp.sources[0].source_url == BASE_URL

    def test_turn_index_increments(self):
        ctx = _make_ctx(results=[_make_search_result()])
        r0 = ctx.chat("assignment")
        r1 = ctx.chat("quiz")
        assert r0.turn_index == 0
        assert r1.turn_index == 1

    def test_matched_keywords_returned(self):
        ctx = _make_ctx()
        resp = ctx.chat("assignment due date")
        assert "assignment" in resp.matched_keywords


# ---------------------------------------------------------------------------
# GooverContext.chat — non-Canvas domain
# ---------------------------------------------------------------------------

class TestGooverContextNonCanvas:
    def test_web_domain_calls_llm(self):
        # web 도메인은 웹 검색 컨텍스트로 LLM을 호출함
        llm = _make_llm()
        ctx = GooverContext(
            retriever=_make_retriever(),
            llm_client=llm,
            llm_model="m",
        )
        ctx.chat("오늘 날씨 어때요?")
        assert llm.chat.completions.create.called

    def test_web_domain_returns_answer(self):
        ctx = _make_ctx()
        resp = ctx.chat("오늘 날씨 어때요?")
        assert resp.answer  # LLM 응답이 있어야 함
        assert resp.domain == "web"

    def test_force_canvas_calls_llm(self):
        llm = _make_llm()
        ctx = GooverContext(
            retriever=_make_retriever([_make_search_result()]),
            llm_client=llm,
            llm_model="m",
        )
        ctx.chat("오늘 날씨?", force_domain="canvas")
        assert llm.chat.completions.create.called


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

class TestGooverContextHistory:
    def test_history_empty_initially(self):
        ctx = _make_ctx()
        assert ctx.history == []

    def test_history_records_turns(self):
        ctx = _make_ctx(results=[_make_search_result()])
        ctx.chat("assignment")
        assert len(ctx.history) == 2
        assert ctx.history[0].role == "user"
        assert ctx.history[1].role == "assistant"

    def test_turn_count_increments(self):
        ctx = _make_ctx(results=[_make_search_result()])
        assert ctx.turn_count == 0
        ctx.chat("assignment")
        assert ctx.turn_count == 1
        ctx.chat("quiz")
        assert ctx.turn_count == 2

    def test_reset_clears_history(self):
        ctx = _make_ctx(results=[_make_search_result()])
        ctx.chat("assignment")
        ctx.reset()
        assert ctx.history == []
        assert ctx.turn_count == 0

    def test_history_included_in_llm_messages(self):
        llm = _make_llm()
        ctx = GooverContext(
            retriever=_make_retriever([_make_search_result()]),
            llm_client=llm,
            llm_model="m",
        )
        ctx.chat("first question about assignment")
        ctx.chat("second question about quiz")
        last_call_messages = llm.chat.completions.create.call_args[1]["messages"]
        # system + at least one prior history turn + current user turn
        assert len(last_call_messages) >= 3

    def test_max_history_turns_respected(self):
        llm = _make_llm()
        ctx = GooverContext(
            retriever=_make_retriever([_make_search_result()]),
            llm_client=llm,
            llm_model="m",
            max_history_turns=2,
        )
        for i in range(5):
            ctx.chat(f"assignment question {i}")
        last_messages = llm.chat.completions.create.call_args[1]["messages"]
        # system(1) + max 2*2 history turns + current(1) = at most 8
        assert len(last_messages) <= 1 + (2 * 2) + 1

    def test_web_turns_also_recorded(self):
        ctx = _make_ctx()
        ctx.chat("오늘 날씨?")
        assert len(ctx.history) == 2
        assert ctx.history[1].content  # LLM 응답이 기록되어야 함

    def test_history_returns_copy(self):
        ctx = _make_ctx(results=[_make_search_result()])
        ctx.chat("assignment")
        h = ctx.history
        h.clear()
        assert len(ctx.history) == 2  # original unaffected


# ---------------------------------------------------------------------------
# build_context factory
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_returns_goover_context(self):
        ctx = build_context(
            _retriever=_make_retriever(),
            _llm_client=_make_llm(),
        )
        assert isinstance(ctx, GooverContext)

    def test_injected_retriever_used(self):
        retriever = _make_retriever()
        ctx = build_context(_retriever=retriever, _llm_client=_make_llm())
        assert ctx._retriever is retriever

    def test_injected_llm_used(self):
        llm = _make_llm()
        ctx = build_context(_retriever=_make_retriever(), _llm_client=llm)
        assert ctx._llm is llm

    def test_max_history_turns_passed(self):
        ctx = build_context(
            max_history_turns=3,
            _retriever=_make_retriever(),
            _llm_client=_make_llm(),
        )
        assert ctx._max_history == 3
