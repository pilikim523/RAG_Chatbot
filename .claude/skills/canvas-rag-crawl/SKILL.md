---
name: canvas-rag-crawl
description: Build or modify the crawler for Instructure Community Canvas Guides. Use when working on URL discovery, HTML fetching, crawl manifests, rate limiting, retries, or crawl safety.
argument-hint: "[task]"
allowed-tools: Read Grep Glob Bash Write Edit
---

# Canvas RAG Crawl Skill

## 목표

Instructure Community `https://community.instructure.com/en/all-guides`에서 Canvas 관련 공식 가이드 URL을 안전하게 수집하고 HTML을 저장한다.

## 작업 원칙

- hostname은 `community.instructure.com`으로 제한한다.
- Canvas 제품 관련 가이드만 1차 수집한다.
- 무제한 mirror crawling 금지.
- concurrency 기본값은 4 이하로 둔다.
- 요청 간 delay 기본값은 800ms 이상으로 둔다.
- 실패 URL은 retry 후 manifest에 상태를 남긴다.
- 모든 산출물은 재시작 가능해야 한다.

## 구현 대상

```text
src/crawler/discover.py
src/crawler/fetch.py
src/crawler/models.py
tests/test_crawler_*.py
```

## 완료 검증

```bash
uv run python -m src.crawler.discover \
  --start-url "https://community.instructure.com/en/all-guides" \
  --product canvas \
  --out data/manifests/canvas_urls.jsonl \
  --dry-run

uv run python -m src.crawler.fetch \
  --manifest data/manifests/canvas_urls.jsonl \
  --out-dir data/raw_html \
  --concurrency 4 \
  --delay-ms 800

uv run pytest -q tests/test_crawler*.py
```
