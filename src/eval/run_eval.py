"""
Canvas RAG golden QA evaluation.

Modes:
  retrieval  — router + retriever only (fast, no LLM)
  answer     — full API end-to-end (slow, needs LLM, uses --answer-sample-n queries)

Usage:
    # Fast retrieval quality (default)
    uv run python -m src.eval.run_eval \\
        --dataset data/eval/canvas_golden_qa.jsonl \\
        --out data/eval/reports/retrieval.json

    # Full end-to-end (answer quality, 5 sample queries)
    uv run python -m src.eval.run_eval \\
        --dataset data/eval/canvas_golden_qa.jsonl \\
        --mode answer \\
        --api-url http://localhost:8080 \\
        --answer-sample-n 5 \\
        --out data/eval/reports/answer.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.eval.metrics import (
    answer_term_rate,
    hit_rate_at_k,
    latency_stats,
    mrr_at_k,
    router_accuracy,
    source_rate,
)

RETRIEVAL_THRESHOLDS = {
    "router_accuracy": 0.90,
    "hit_rate@5": 0.80,
    "source_rate": 0.95,
    "p95_latency_ms": 5000.0,    # retrieval-only: ~30-70ms warm
}
ANSWER_THRESHOLDS = {
    "router_accuracy": 0.90,
    "hit_rate@5": 0.80,
    "source_rate": 0.95,
    "answer_term_rate": 0.75,
    "p95_latency_ms": 60000.0,   # local 7B LLM: 30-60s expected
}


# ---------------------------------------------------------------------------
# Retrieval-only mode (no LLM)
# ---------------------------------------------------------------------------

def _run_retrieval(items: list[dict]) -> tuple[list[dict], list[float]]:
    from src.retrieval.retriever import get_retriever
    from src.retrieval.router import DomainRouter

    retriever = get_retriever()
    router = DomainRouter()
    # warm-up: trigger model load so cold-start doesn't skew p95 latency
    retriever.search("Canvas warmup query", top_k=1)
    results: list[dict] = []
    latencies: list[float] = []

    for item in items:
        qid = item["id"]
        try:
            t0 = time.perf_counter()
            decision = router.route(item["query"])
            search_results = []
            if decision.is_canvas:
                role = item.get("role")
                search_results = retriever.search(item["query"], top_k=5, role=role)
                # role fallback
                if not search_results and role:
                    search_results = retriever.search(item["query"], top_k=5)
            lat_ms = (time.perf_counter() - t0) * 1000

            result = dict(item)
            result["actual_domain"] = decision.domain
            result["sources"] = [
                {"source_url": r.source_url, "title": r.title, "score": r.score}
                for r in search_results
            ]
            result["answer"] = None
            result["latency_ms"] = lat_ms
            results.append(result)
            latencies.append(lat_ms)
            status = "✓" if result["actual_domain"] == item["expected_domain"] else "✗"
            print(f"  {status} [{qid}] domain={result['actual_domain']} sources={len(result['sources'])} lat={lat_ms:.0f}ms")
        except Exception as exc:
            print(f"  ! [{qid}] ERROR: {exc}")
    return results, latencies


# ---------------------------------------------------------------------------
# Answer mode (full API)
# ---------------------------------------------------------------------------

def _call_api(api_url: str, item: dict) -> tuple[dict, float]:
    payload: dict = {"query": item["query"], "top_k": 5}
    if item.get("role"):
        payload["role"] = item["role"]
    if item.get("force_domain"):
        payload["force_domain"] = item["force_domain"]
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{api_url}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read())
    return body, (time.perf_counter() - t0) * 1000


def _run_answer(items: list[dict], api_url: str, sample_n: int) -> tuple[list[dict], list[float]]:
    canvas_items = [i for i in items if i.get("expected_domain") == "canvas"][:sample_n]
    results: list[dict] = []
    latencies: list[float] = []
    errors: list[str] = []

    for item in canvas_items:
        qid = item["id"]
        try:
            resp, lat_ms = _call_api(api_url, item)
            result = dict(item)
            result["actual_domain"] = resp.get("domain", "unknown")
            result["answer"] = resp.get("answer", "")
            result["sources"] = resp.get("sources", [])
            result["latency_ms"] = lat_ms
            results.append(result)
            latencies.append(lat_ms)
            status = "✓" if result["actual_domain"] == item["expected_domain"] else "✗"
            print(f"  {status} [{qid}] domain={result['actual_domain']} sources={len(result['sources'])} lat={lat_ms:.0f}ms")
        except Exception as exc:
            errors.append(f"{qid}: {exc}")
            print(f"  ! [{qid}] ERROR: {exc}")
    return results, latencies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    dataset_path: str,
    mode: str,
    api_url: str,
    out_path: str,
    answer_sample_n: int,
) -> dict:
    items = [
        json.loads(line)
        for line in Path(dataset_path).read_text().splitlines()
        if line.strip()
    ]
    print(f"Mode: {mode} | {len(items)} queries")

    if mode == "retrieval":
        results, latencies = _run_retrieval(items)
    else:
        results, latencies = _run_answer(items, api_url, answer_sample_n)

    lat = latency_stats(latencies)
    metrics: dict = {
        "router_accuracy": router_accuracy(results),
        "hit_rate@5": hit_rate_at_k(results, k=5),
        "mrr@5": mrr_at_k(results, k=5),
        "source_rate": source_rate(results),
        "p50_latency_ms": lat["p50"],
        "p95_latency_ms": lat["p95"],
        "mean_latency_ms": lat["mean"],
    }
    if mode == "answer":
        metrics["answer_term_rate"] = answer_term_rate(results)

    thresholds = ANSWER_THRESHOLDS if mode == "answer" else RETRIEVAL_THRESHOLDS
    passed = {
        k: (metrics.get(k, 0) >= v) if k != "p95_latency_ms" else (metrics.get(k, 9999) <= v)
        for k, v in thresholds.items()
    }

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dataset": dataset_path,
        "total": len(items),
        "evaluated": len(results),
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "overall_pass": all(passed.values()),
        "results": results,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n=== Evaluation Report ===")
    for k, v in metrics.items():
        if k in thresholds:
            flag = "PASS" if passed[k] else "FAIL"
            print(f"  {flag}  {k}: {v:.3f}  (threshold: {thresholds[k]})")
        else:
            print(f"       {k}: {v:.3f}")
    print(f"\nOverall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"Report: {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/canvas_golden_qa.jsonl")
    parser.add_argument("--mode", choices=["retrieval", "answer"], default="retrieval")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--answer-sample-n", type=int, default=5)
    parser.add_argument("--out", default="data/eval/reports/latest.json")
    args = parser.parse_args()
    run(args.dataset, args.mode, args.api_url, args.out, args.answer_sample_n)


if __name__ == "__main__":
    main()
