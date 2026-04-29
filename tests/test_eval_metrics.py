"""Unit tests for src/eval/metrics.py — no API calls required."""
from __future__ import annotations

import pytest

from src.eval.metrics import (
    answer_term_rate,
    hit_rate_at_k,
    latency_stats,
    mrr_at_k,
    router_accuracy,
    source_rate,
)


def _canvas_result(
    actual_domain: str = "canvas",
    expected_domain: str = "canvas",
    sources: list | None = None,
    answer: str = "assignment를 제출하세요",
    required_answer_terms: list | None = None,
    relevant_url_patterns: list | None = None,
    min_sources: int = 1,
) -> dict:
    return {
        "actual_domain": actual_domain,
        "expected_domain": expected_domain,
        "sources": [{"source_url": "https://community.instructure.com/en/kb/articles/661210-submit", "score": 0.9}] if sources is None else sources,
        "answer": answer,
        "required_answer_terms": ["assignment"] if required_answer_terms is None else required_answer_terms,
        "relevant_url_patterns": ["submit"] if relevant_url_patterns is None else relevant_url_patterns,
        "min_sources": min_sources,
    }


def _general_result(actual_domain: str = "general") -> dict:
    return {
        "actual_domain": actual_domain,
        "expected_domain": "general",
        "sources": [],
        "answer": "날씨 정보를 제공할 수 없습니다.",
        "required_answer_terms": [],
        "relevant_url_patterns": [],
        "min_sources": 0,
    }


# ---------------------------------------------------------------------------
# router_accuracy
# ---------------------------------------------------------------------------

class TestRouterAccuracy:
    def test_all_correct(self):
        results = [
            _canvas_result(actual_domain="canvas", expected_domain="canvas"),
            _general_result(actual_domain="general"),
        ]
        assert router_accuracy(results) == 1.0

    def test_all_wrong(self):
        results = [
            _canvas_result(actual_domain="general", expected_domain="canvas"),
        ]
        assert router_accuracy(results) == 0.0

    def test_partial(self):
        results = [
            _canvas_result(actual_domain="canvas", expected_domain="canvas"),
            _canvas_result(actual_domain="general", expected_domain="canvas"),
        ]
        assert router_accuracy(results) == 0.5

    def test_empty_returns_zero(self):
        assert router_accuracy([]) == 0.0


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------

class TestHitRateAtK:
    def test_hit_when_pattern_matches(self):
        r = _canvas_result(
            sources=[{"source_url": "https://example.com/submit-assignment", "score": 0.9}],
            relevant_url_patterns=["submit"],
        )
        assert hit_rate_at_k([r]) == 1.0

    def test_miss_when_no_pattern_match(self):
        r = _canvas_result(
            sources=[{"source_url": "https://example.com/calendar", "score": 0.9}],
            relevant_url_patterns=["submit"],
        )
        assert hit_rate_at_k([r]) == 0.0

    def test_no_patterns_hit_if_sources_present(self):
        r = _canvas_result(
            sources=[{"source_url": "https://example.com/something", "score": 0.8}],
            relevant_url_patterns=[],
        )
        assert hit_rate_at_k([r]) == 1.0

    def test_no_patterns_miss_if_no_sources(self):
        r = _canvas_result(sources=[], relevant_url_patterns=[])
        assert hit_rate_at_k([r]) == 0.0

    def test_ignores_non_canvas_results(self):
        results = [_canvas_result(), _general_result()]
        assert hit_rate_at_k(results) == 1.0

    def test_only_top_k_considered(self):
        r = _canvas_result(
            sources=[
                {"source_url": "https://example.com/other", "score": 0.9},
                {"source_url": "https://example.com/submit", "score": 0.8},
            ],
            relevant_url_patterns=["submit"],
        )
        # k=1 → only first source checked, no match
        assert hit_rate_at_k([r], k=1) == 0.0
        # k=2 → second source matches
        assert hit_rate_at_k([r], k=2) == 1.0

    def test_empty_results_returns_zero(self):
        assert hit_rate_at_k([]) == 0.0


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------

class TestMrrAtK:
    def test_first_position_gives_one(self):
        r = _canvas_result(
            sources=[{"source_url": "https://example.com/submit", "score": 0.9}],
            relevant_url_patterns=["submit"],
        )
        assert mrr_at_k([r]) == pytest.approx(1.0)

    def test_second_position_gives_half(self):
        r = _canvas_result(
            sources=[
                {"source_url": "https://example.com/calendar", "score": 0.9},
                {"source_url": "https://example.com/submit", "score": 0.8},
            ],
            relevant_url_patterns=["submit"],
        )
        assert mrr_at_k([r]) == pytest.approx(0.5)

    def test_no_match_gives_zero(self):
        r = _canvas_result(
            sources=[{"source_url": "https://example.com/calendar", "score": 0.9}],
            relevant_url_patterns=["submit"],
        )
        assert mrr_at_k([r]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert mrr_at_k([]) == 0.0


# ---------------------------------------------------------------------------
# source_rate
# ---------------------------------------------------------------------------

class TestSourceRate:
    def test_sources_present(self):
        r = _canvas_result(sources=[{"source_url": "https://example.com/a", "score": 0.8}], min_sources=1)
        assert source_rate([r]) == 1.0

    def test_no_sources_fails(self):
        r = _canvas_result(sources=[], min_sources=1)
        assert source_rate([r]) == 0.0

    def test_general_queries_excluded(self):
        results = [_canvas_result(sources=[]), _general_result()]
        assert source_rate(results) == 0.0

    def test_empty_returns_zero(self):
        assert source_rate([]) == 0.0


# ---------------------------------------------------------------------------
# answer_term_rate
# ---------------------------------------------------------------------------

class TestAnswerTermRate:
    def test_all_terms_present(self):
        r = _canvas_result(
            answer="Canvas의 assignment를 제출하는 방법입니다.",
            required_answer_terms=["assignment", "제출"],
        )
        assert answer_term_rate([r]) == 1.0

    def test_missing_term_fails(self):
        r = _canvas_result(
            answer="Canvas에서 파일을 업로드합니다.",
            required_answer_terms=["assignment", "제출"],
        )
        assert answer_term_rate([r]) == 0.0

    def test_case_insensitive(self):
        r = _canvas_result(
            answer="Use the ASSIGNMENT page to submit.",
            required_answer_terms=["assignment"],
        )
        assert answer_term_rate([r]) == 1.0

    def test_empty_terms_always_passes(self):
        r = _canvas_result(answer="어떤 답변", required_answer_terms=[])
        assert answer_term_rate([r]) == 1.0

    def test_empty_results_returns_zero(self):
        assert answer_term_rate([]) == 0.0


# ---------------------------------------------------------------------------
# latency_stats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_basic(self):
        stats = latency_stats([100.0, 200.0, 300.0, 400.0, 500.0])
        assert stats["p50"] == 300.0
        assert stats["p95"] == 500.0
        assert stats["mean"] == pytest.approx(300.0)

    def test_single_value(self):
        stats = latency_stats([250.0])
        assert stats["p50"] == 250.0
        assert stats["p95"] == 250.0
        assert stats["mean"] == 250.0

    def test_empty(self):
        stats = latency_stats([])
        assert stats["p50"] == 0.0
        assert stats["p95"] == 0.0
