---
name: canvas-rag-eval
description: Build and run retrieval/answer quality evaluation for the Canvas Guides chatbot.
argument-hint: "[task]"
allowed-tools: Read Grep Glob Bash Write Edit
---

# Canvas RAG Eval Skill

## 목표

Canvas RAG 챗봇이 근거 기반으로 정확하게 답하는지 평가한다.

## 평가 원칙

- retrieval 품질과 answer 품질을 분리한다.
- Canvas 공식 문서 근거가 없는 답변은 실패로 본다.
- 답변이 맞아 보여도 source가 없으면 실패로 본다.
- route 오탐/미탐을 별도 측정한다.
- golden dataset은 role별로 구성한다.

## 핵심 지표

```text
router_accuracy
retrieval_hit_rate@5
retrieval_mrr@5
source_required_answer_rate
no_source_hallucination_rate
answer_contains_required_terms
p50_latency
p95_latency
```

## 최소 배포 기준

```text
router_accuracy >= 0.90
retrieval_hit_rate@5 >= 0.80
source_required_answer_rate >= 0.95
no_source_hallucination_rate >= 0.98
p95_chat_latency <= 5s
```

## 평가 실행

```bash
uv run python -m src.eval.run_eval \
  --dataset data/eval/canvas_golden_qa.jsonl \
  --collection canvas_guides \
  --out data/eval/reports/latest.json
```
