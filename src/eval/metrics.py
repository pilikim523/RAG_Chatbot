"""Pure metric computation functions for Canvas RAG evaluation."""
from __future__ import annotations

import statistics
from typing import Any


def router_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(
        1 for r in results
        if r.get("actual_domain") == r.get("expected_domain")
    )
    return correct / len(results)


def hit_rate_at_k(results: list[dict], k: int = 5) -> float:
    """Fraction of canvas queries with at least one relevant source in top-k.

    Items with skip_hit=True are excluded (topic not in index).
    """
    canvas = [
        r for r in results
        if r.get("expected_domain") == "canvas" and not r.get("skip_hit")
    ]
    if not canvas:
        return 0.0
    hits = 0
    for r in canvas:
        patterns = r.get("relevant_url_patterns", [])
        sources = r.get("sources", [])[:k]
        if not patterns:
            if sources:
                hits += 1
            continue
        for src in sources:
            url = src.get("source_url", "")
            if any(p in url for p in patterns):
                hits += 1
                break
    return hits / len(canvas)


def mrr_at_k(results: list[dict], k: int = 5) -> float:
    """Mean Reciprocal Rank over canvas queries (skip_hit items excluded)."""
    canvas = [
        r for r in results
        if r.get("expected_domain") == "canvas" and not r.get("skip_hit")
    ]
    if not canvas:
        return 0.0
    rrs = []
    for r in canvas:
        patterns = r.get("relevant_url_patterns", [])
        sources = r.get("sources", [])[:k]
        if not patterns:
            rrs.append(1.0 if sources else 0.0)
            continue
        rr = 0.0
        for i, src in enumerate(sources, start=1):
            if any(p in src.get("source_url", "") for p in patterns):
                rr = 1.0 / i
                break
        rrs.append(rr)
    return statistics.mean(rrs)


def source_rate(results: list[dict]) -> float:
    """Fraction of canvas queries that returned >= min_sources sources."""
    canvas = [r for r in results if r.get("expected_domain") == "canvas"]
    if not canvas:
        return 0.0
    ok = sum(
        1 for r in canvas
        if len(r.get("sources", [])) >= max(r.get("min_sources", 1), 1)
    )
    return ok / len(canvas)


def answer_term_rate(results: list[dict]) -> float:
    """Fraction of canvas queries where the answer contains all required terms."""
    canvas = [r for r in results if r.get("expected_domain") == "canvas"]
    if not canvas:
        return 0.0
    ok = 0
    for r in canvas:
        terms = r.get("required_answer_terms", [])
        answer = (r.get("answer") or "").lower()
        if all(t.lower() in answer for t in terms):
            ok += 1
    return ok / len(canvas)


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    s = sorted(latencies_ms)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(int(n * p / 100), n - 1)]

    return {
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.mean(s),
    }
