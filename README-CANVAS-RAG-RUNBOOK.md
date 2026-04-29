# Canvas Guides RAG 챗봇 운영 Runbook

## 아키텍처 요약

```text
Instructure Community All Guides
  → src/crawler/discover.py   (URL 수집)
  → src/crawler/fetch.py      (HTML fetch)
  → src/cleaner/to_markdown.py (Markdown 정제)
  → src/indexing/chunk.py     (heading-aware 청킹)
  → src/indexing/qdrant_index.py (BAAI/bge-m3 MPS 임베딩 → Qdrant)
  → src/retrieval/retriever.py (벡터 검색)
  → src/retrieval/router.py   (Canvas 도메인 라우팅)
  → src/api/chat.py           (ChatHandler → Ollama LLM → 한국어 답변)
  → src/api/main.py           (FastAPI POST /chat)
```

| 컴포넌트 | 실행 위치 | 비고 |
|---|---|---|
| Qdrant | Docker (OrbStack/Docker Desktop) | `docker-compose.macos.yml` |
| Embedding (BAAI/bge-m3) | macOS host, MPS | batch_size=8 (OOM 방지) |
| LLM (qwen2.5:7b-instruct) | macOS host, Ollama + Metal | |
| FastAPI | macOS host | `uvicorn src.api.main:app` |

---

## 1. 초기 설치

```bash
# 시스템 도구
brew install uv jq ollama

# Python 의존성
cd canvas-rag-chatbot
uv sync

# ML/embedding 그룹 (sentence-transformers, torch)
uv sync --group embedding
```

---

## 2. 서비스 시작

```bash
# Qdrant (OrbStack 또는 Docker Desktop)
docker compose -f docker-compose.macos.yml up -d qdrant
curl http://localhost:6333/readyz

# Ollama
ollama serve &
ollama pull qwen2.5:7b-instruct

# FastAPI
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

---

## 3. 데이터 파이프라인 (최초 실행 또는 재수집)

```bash
# Step 1: URL discovery
uv run python -m src.crawler.discover \
  --start-url "https://community.instructure.com/en/all-guides" \
  --product canvas \
  --out data/manifests/canvas_urls.jsonl

# Step 2: HTML fetch
uv run python -m src.crawler.fetch \
  --manifest data/manifests/canvas_urls.jsonl \
  --out-dir data/raw_html \
  --concurrency 4 \
  --delay-ms 800

# Step 3: Markdown 변환
uv run python -m src.cleaner.to_markdown \
  --input-manifest data/manifests/canvas_urls.jsonl \
  --raw-html-dir data/raw_html \
  --out-dir data/markdown \
  --out-manifest data/manifests/canvas_docs.jsonl

# Step 4: 청킹
uv run python -m src.indexing.chunk \
  --manifest data/manifests/canvas_docs.jsonl \
  --out data/chunks/canvas_chunks.jsonl \
  --chunk-size 900 \
  --chunk-overlap 120

# Step 5: Qdrant 적재 (멱등, content_hash 기반 중복 스킵)
uv run python -m src.indexing.qdrant_index \
  --chunks data/chunks/canvas_chunks.jsonl \
  --collection canvas_guides \
  --qdrant-url http://localhost:6333 \
  --embedder auto
```

> bge-m3 첫 로드: ~7초. 이후 청크당 ~30ms (MPS warm).

---

## 4. Smoke test

```bash
# Retriever 직접 테스트
uv run python -m src.retrieval.smoke_test \
  --query "Canvas에서 assignment 제출하는 방법" \
  --top-k 5

# API 엔드포인트 테스트
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Canvas에서 과제 due date와 availability date 차이를 알려줘",
    "role": "instructor"
  }'
```

---

## 5. Golden QA 평가

```bash
# 빠른 retrieval 품질 평가 (LLM 없음, ~60초)
uv run python -m src.eval.run_eval \
  --dataset data/eval/canvas_golden_qa.jsonl \
  --mode retrieval \
  --out data/eval/reports/retrieval.json

# 전체 answer 품질 평가 (LLM 포함, 샘플 5개, ~3분)
uv run python -m src.eval.run_eval \
  --dataset data/eval/canvas_golden_qa.jsonl \
  --mode answer \
  --api-url http://localhost:8080 \
  --answer-sample-n 5 \
  --out data/eval/reports/answer.json
```

최소 통과 기준:

| 지표 | 기준 |
|---|---|
| router_accuracy | ≥ 0.90 |
| hit_rate@5 | ≥ 0.80 |
| source_rate | ≥ 0.95 |
| answer_term_rate | ≥ 0.75 |
| p95_latency (retrieval) | ≤ 5s |
| p95_latency (answer/LLM) | ≤ 60s |

---

## 6. 테스트

```bash
uv run pytest tests/ -q
# 예상: 350 passed
```

---

## 7. 운영 배포 전 체크리스트

- [x] Qdrant canvas_guides 컬렉션 667 포인트 적재 확인
- [x] bge-m3 MPS embedding smoke test 통과
- [x] Ollama qwen2.5:7b-instruct Metal 실행 확인
- [x] FastAPI /chat end-to-end 한국어 답변 + 출처 URL 확인
- [x] golden QA retrieval PASS (router 1.0, hit_rate 1.0, source_rate 1.0)
- [x] golden QA answer PASS (answer_term_rate 0.80)
- [ ] Qdrant collection snapshot 백업
- [ ] API_KEY 설정 (운영 환경에서는 필수)
- [ ] 사내 인증 미들웨어 적용
- [ ] PII 마스킹 로그 적용
- [ ] Canvas 문서 증분 재수집 주기 설정 (월 1회 권장)
- [ ] Canvas 미수집 항목 크롤 (SpeedGrader, Quiz 문제, Syllabus, 학생 등록)

---

## 8. 알려진 제약

| 항목 | 현황 | 해결 방법 |
|---|---|---|
| Canvas 미수집 문서 | SpeedGrader, Quiz 생성, Syllabus, 학생 등록 가이드 없음 | 크롤 범위 확장 후 재인덱싱 |
| Role 필터 | 인덱스에 `student` role 없음 (None/instructor만 존재) | ChatHandler가 자동 fallback으로 role 없이 재검색 |
| LLM 응답 지연 | 로컬 7B 모델 평균 30초 | 운영 환경에서는 OpenAI API로 교체 가능 (LLM_PROVIDER=openai) |
| MPS bge-m3 OOM | batch_size=64 이상 시 21GB 버퍼 오류 | EMBEDDING_BATCH_SIZE=8 유지 |
