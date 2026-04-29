"""Tests for src/retrieval/router.py"""
from __future__ import annotations

import pytest

from src.retrieval.router import (
    DomainRouter,
    RouteDecision,
    _detect_canvas_keywords,
    route_query,
)


# ---------------------------------------------------------------------------
# _detect_canvas_keywords
# ---------------------------------------------------------------------------

class TestDetectCanvasKeywords:
    def test_en_keyword_canvas(self):
        assert "canvas" in _detect_canvas_keywords("How do I use Canvas?")

    def test_en_keyword_assignment(self):
        assert "assignment" in _detect_canvas_keywords("Where is my assignment?")

    def test_en_keyword_gradebook(self):
        assert "gradebook" in _detect_canvas_keywords("How do I open the gradebook?")

    def test_en_keyword_quiz(self):
        assert "quiz" in _detect_canvas_keywords("Create a new quiz")

    def test_en_keyword_lti(self):
        assert "lti" in _detect_canvas_keywords("Enable LTI integration")

    def test_en_keyword_sis(self):
        assert "sis" in _detect_canvas_keywords("SIS import failed")

    def test_en_keyword_speedgrader(self):
        assert "speedgrader" in _detect_canvas_keywords("open SpeedGrader")

    def test_ko_keyword_과제(self):
        assert "과제" in _detect_canvas_keywords("과제를 어떻게 제출하나요?")

    def test_ko_keyword_성적(self):
        assert "성적" in _detect_canvas_keywords("성적 확인 방법이 궁금합니다")

    def test_ko_keyword_학생(self):
        assert "학생" in _detect_canvas_keywords("학생 등록 방법")

    def test_ko_keyword_교수자(self):
        assert "교수자" in _detect_canvas_keywords("교수자 권한 설정")

    def test_ko_keyword_관리자(self):
        assert "관리자" in _detect_canvas_keywords("관리자 메뉴 찾기")

    def test_ko_keyword_퀴즈(self):
        assert "퀴즈" in _detect_canvas_keywords("퀴즈 만드는 법")

    def test_ko_keyword_강좌(self):
        assert "강좌" in _detect_canvas_keywords("강좌 개설 방법")

    def test_empty_query_returns_empty(self):
        assert _detect_canvas_keywords("") == []

    def test_irrelevant_query_returns_empty(self):
        result = _detect_canvas_keywords("오늘 점심 뭐 먹을까")
        assert result == []

    def test_case_insensitive_en(self):
        assert "assignment" in _detect_canvas_keywords("ASSIGNMENT due date")

    def test_deduplicates_keywords(self):
        kws = _detect_canvas_keywords("assignment과 assignment 제출")
        assert kws.count("assignment") == 1

    def test_mixed_en_ko(self):
        kws = _detect_canvas_keywords("Canvas에서 과제 제출하는 방법")
        assert "canvas" in kws
        assert "과제" in kws

    def test_word_boundary_prevents_partial_match(self):
        # "grad" should not match "gradebook"
        kws = _detect_canvas_keywords("undergraduate student")
        # "student" should match, "undergraduate" should not trigger spurious matches
        assert "student" in kws

    def test_multiple_keywords_all_returned(self):
        kws = _detect_canvas_keywords("How do I grade a quiz in Canvas?")
        assert "canvas" in kws
        assert "quiz" in kws


# ---------------------------------------------------------------------------
# RouteDecision
# ---------------------------------------------------------------------------

class TestRouteDecision:
    def test_is_canvas_true(self):
        decision = RouteDecision(domain="canvas", matched_keywords=["assignment"])
        assert decision.is_canvas is True

    def test_is_canvas_false_for_general(self):
        assert RouteDecision(domain="general").is_canvas is False

    def test_forced_defaults_to_false(self):
        assert RouteDecision(domain="canvas").forced is False

    def test_matched_keywords_default_empty(self):
        assert RouteDecision(domain="general").matched_keywords == []


# ---------------------------------------------------------------------------
# DomainRouter.route
# ---------------------------------------------------------------------------

class TestDomainRouter:
    router = DomainRouter()

    # Canvas detection
    def test_canvas_keyword_routes_to_canvas(self):
        d = self.router.route("How do I submit an assignment?")
        assert d.domain == "canvas"
        assert d.is_canvas

    def test_canvas_literal_routes_to_canvas(self):
        d = self.router.route("Canvas 로그인이 안 돼요")
        assert d.domain == "canvas"

    def test_ko_canvas_keyword_routes_to_canvas(self):
        d = self.router.route("과제 제출 기한이 지났어요")
        assert d.domain == "canvas"

    def test_matched_keywords_populated(self):
        d = self.router.route("assignment submission help")
        assert len(d.matched_keywords) > 0

    # General fallback
    def test_unrelated_query_routes_to_general(self):
        d = self.router.route("오늘 날씨 어때요?")
        assert d.domain == "general"

    def test_empty_query_routes_to_general(self):
        d = self.router.route("")
        assert d.domain == "general"

    # force_domain
    def test_force_canvas_overrides_detection(self):
        d = self.router.route("오늘 날씨 어때요?", force_domain="canvas")
        assert d.domain == "canvas"
        assert d.forced is True

    def test_force_general_overrides_canvas_keywords(self):
        d = self.router.route("assignment submission", force_domain="general")
        assert d.domain == "general"
        assert d.forced is True

    def test_force_internal(self):
        d = self.router.route("사내 정책 문의", force_domain="internal")
        assert d.domain == "internal"
        assert d.forced is True

    def test_forced_decision_has_no_keywords(self):
        d = self.router.route("assignment", force_domain="canvas")
        # forced decision skips keyword detection
        assert d.matched_keywords == []


# ---------------------------------------------------------------------------
# route_query convenience wrapper
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def test_canvas_query(self):
        assert route_query("How do I use gradebook?").domain == "canvas"

    def test_general_query(self):
        assert route_query("내일 회의 몇 시에요?").domain == "general"

    def test_force_domain(self):
        d = route_query("anything", force_domain="canvas")
        assert d.forced is True
