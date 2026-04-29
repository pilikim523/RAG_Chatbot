# CLAUDE.md

## 프로젝트 목적

이 저장소는 Instructure Community `All Guides`에서 Canvas 관련 공식 가이드를 수집, Markdown 정제, 청킹, 임베딩, Qdrant 적재 후 사내 업무 연관자에게 챗봇으로 제공하는 RAG 시스템이다.

최종 목표는 다음이다.

```text
Canvas 관련 질문
  -> internal Canvas RAG 우선 검색
  -> 근거 기반 한국어 답변
  -> 공식 문서 출처 제공
```

---

## 개발 환경

- 개발 장비: Apple Silicon M3 Pro MacBook
- 운영 개발 방식:
  - Qdrant: Docker Desktop으로 실행 가능
  - Embedding/LLM local runtime: macOS 네이티브 실행 우선
  - Apple GPU 사용 경로:
    - Ollama: Metal API 기반 GPU acceleration
    - MLX: Apple Silicon unified memory 최적화
    - PyTorch/SentenceTransformers: MPS backend 사용 가능
- CUDA 전제 코드는 작성하지 않는다.
- Docker 컨테이너 내부에서 Apple GPU를 쓰려는 구조는 기본값으로 사용하지 않는다.
- Qdrant만 컨테이너로 실행하고, 로컬 임베딩/LLM은 host macOS에서 실행한다.

---

## 절대 원칙

- Canvas 관련 답변은 LLM 일반지식보다 Qdrant `canvas_guides` 컬렉션 검색 결과를 우선한다.
- 검색 근거가 없으면 추측하지 않는다. `현재 수집된 Canvas 공식 문서에서 확인된 근거가 없습니다.`로 처리한다.
- 모든 Canvas 답변에는 최소 1개 이상의 `source_url`, `title`을 포함한다.
- 사내 정책, 학교별 운영 방식, 계정 권한, SIS/LTI 연동 설정은 공식 Canvas 문서와 다를 수 있으므로 내부 운영 문서가 있으면 내부 문서를 더 높은 우선순위로 둔다.
- 크롤러는 대상 도메인과 경로를 제한하고, rate limit, retry, content hash, manifest 기록을 반드시 구현한다.
- 인덱싱은 멱등성을 보장한다. 같은 `source_url + content_hash`는 중복 적재하지 않는다.
- 사용자 질문은 기본 한국어로 답변하되 Canvas 공식 UI 용어는 영어 원문을 병기한다.
- 운영 API는 인증 없는 공개 엔드포인트로 만들지 않는다.
- `.env`, API Key, 토큰, 쿠키, 세션, 사내 사용자 식별정보는 로그에 남기지 않는다.
- 테스트 없이 완료로 보고하지 않는다.

---

## 기본 아키텍처

```text
Instructure All Guides
  -> crawler/discover
  -> crawler/fetch
  -> cleaner/markdown
  -> chunker
  -> embedding
  -> Qdrant canvas_guides
  -> retriever
  -> domain router
  -> chatbot_goover_context.py
  -> FastAPI API
  -> 사내 챗봇 UI
```

---

## M3 Pro 우선 런타임 정책

### 1순위: 로컬 임베딩은 Apple GPU 사용

기본 embedding provider 우선순위:

```text
1. local_mps_sentence_transformers
2. local_mlx_embedding
3. openai_embedding_api
```

권장 개발 기본값:

```bash
EMBEDDING_PROVIDER=local_mps
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
LOCAL_EMBEDDING_DEVICE=mps
```

대량 인덱싱에서 MPS/MLX가 불안정하면 OpenAI embedding으로 fallback 가능하되, fallback 여부를 로그에 남긴다.

### 2순위: 로컬 LLM은 Ollama host 실행

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

M3 Pro unified memory가 18GB이면 7B/8B 계열을 기본값으로 둔다.
M3 Pro unified memory가 36GB 이상이면 14B 계열을 테스트할 수 있다.
35B 이상 모델은 기본 개발값으로 두지 않는다.

### 3순위: Qdrant는 Docker

```bash
docker compose -f docker-compose.macos.yml up -d qdrant
```

---

## 권장 저장소 구조

```text
canvas-rag-chatbot/
├── CLAUDE.md
├── README-CANVAS-RAG-RUNBOOK.md
├── RUNBOOK-M3PRO-MAC-GPU.md
├── docker-compose.macos.yml
├── pyproject.toml
├── .env.example
├── .claude/
│   ├── settings.json
│   ├── hooks/
│   │   ├── protect-files.sh
│   │   ├── post-edit-format.sh
│   │   ├── validate-no-secrets.sh
│   │   └── verify-macos-gpu.sh
│   ├── skills/
│   │   ├── canvas-rag-crawl/SKILL.md
│   │   ├── canvas-rag-ingest/SKILL.md
│   │   ├── canvas-rag-chatbot/SKILL.md
│   │   └── canvas-rag-eval/SKILL.md
│   └── agents/
│       ├── crawler-engineer.md
│       ├── rag-indexing-engineer.md
│       ├── chatbot-api-engineer.md
│       ├── rag-qa-evaluator.md
│       └── macos-mlx-runtime-engineer.md
├── data/
│   ├── raw_html/
│   ├── markdown/
│   ├── chunks/
│   ├── manifests/
│   └── eval/
├── src/
│   ├── crawler/
│   ├── cleaner/
│   ├── indexing/
│   ├── retrieval/
│   ├── api/
│   └── chatbot_goover_context.py
└── tests/
```

---

## 주요 명령어

```bash
# macOS 개발 의존성
brew install uv jq tree qdrant ollama

# Python 의존성
uv sync

# Qdrant 실행
docker compose -f docker-compose.macos.yml up -d qdrant

# Ollama host 실행
ollama serve

# 개발용 로컬 모델
ollama pull qwen2.5:7b-instruct

# URL discovery
uv run python -m src.crawler.discover \
  --start-url "https://community.instructure.com/en/all-guides" \
  --product canvas \
  --out data/manifests/canvas_urls.jsonl

# HTML fetch
uv run python -m src.crawler.fetch \
  --manifest data/manifests/canvas_urls.jsonl \
  --out-dir data/raw_html \
  --concurrency 4 \
  --delay-ms 800

# Markdown 변환
uv run python -m src.cleaner.to_markdown \
  --input-manifest data/manifests/canvas_urls.jsonl \
  --raw-html-dir data/raw_html \
  --out-dir data/markdown \
  --out-manifest data/manifests/canvas_docs.jsonl

# Chunk 생성
uv run python -m src.indexing.chunk \
  --manifest data/manifests/canvas_docs.jsonl \
  --out data/chunks/canvas_chunks.jsonl \
  --chunk-size 900 \
  --chunk-overlap 120

# Qdrant 적재
uv run python -m src.indexing.qdrant_index \
  --chunks data/chunks/canvas_chunks.jsonl \
  --collection canvas_guides \
  --qdrant-url http://localhost:6333

# Heavy ML embedding group (sentence-transformers + torch)
uv sync --group embedding

# Retriever smoke test
uv run python -m src.retrieval.smoke_test \
  --query "Canvas에서 assignment due date와 availability date 차이는?"

# API 실행
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

---

## Canvas 문서 수집 규칙

- 시작 URL은 `https://community.instructure.com/en/all-guides`이다.
- 제품 범위는 1차로 Canvas 관련 가이드만 수집한다.
- Mastery, Parchment, Elevate, LearnPlatform 등은 Canvas 검색 품질을 떨어뜨리지 않도록 별도 컬렉션으로 분리한다.
- URL discovery는 링크 그래프를 따라가되 hostname은 `community.instructure.com`으로 제한한다.
- `source_url`, `canonical_url`, `title`, `guide`, `category`, `role`, `crawled_at`, `content_hash`를 manifest에 기록한다.
- 삭제/변경 감지를 위해 fetch 결과와 markdown 결과의 hash를 모두 저장한다.
- 장애 재시작을 위해 이미 성공한 URL은 skip 가능해야 한다.

---

## Qdrant 컬렉션 규칙

- 컬렉션명: `canvas_guides`
- distance: cosine 또는 dot. embedding provider 기준에 맞춘다.
- payload index:
  - `source_url`
  - `guide`
  - `category`
  - `role`
  - `product`
  - `content_hash`
- 검색 시 role/category 필터를 지원한다.
- 적재는 batch upsert를 사용한다.
- 재색인 시 전체 recreate보다 hash 기반 incremental update를 우선한다.

---

## Domain Routing 규칙

`src/retrieval/router.py`에 다음 정책을 구현한다.

- `canvas` 명시어가 있으면 Canvas RAG로 라우팅한다.
- 다음 키워드는 Canvas 후보로 본다.
  - course, assignment, gradebook, quiz, module, rubric, speedgrader
  - discussion, announcement, page, syllabus, calendar, section
  - enrollment, role, permission, admin, instructor, student, observer
  - LTI, SIS, Canvas Studio, Commons, Catalog
  - 과제, 성적, 퀴즈, 모듈, 루브릭, 강의, 학생, 교수자, 관리자
- UI에서 domain이 선택된 경우 사용자 선택을 우선한다.
- Canvas 후보인데 검색 결과가 없으면 일반 LLM 답변으로 fallback하지 않는다.
- Canvas가 아닌 사내 정책/운영 질문은 내부 문서 RAG가 있으면 내부 문서를 먼저 검색한다.

---

## `chatbot_goover_context.py` 연결 규칙

- 기존 파일을 먼저 읽고 인터페이스를 깨지 않는다.
- 새 retriever는 DI 또는 factory로 주입한다.
- Canvas routing 결과와 retrieved context를 `GooverContext` 또는 기존 context payload에 추가한다.
- LLM 호출부에는 다음 system rule을 포함한다.

```text
Canvas 관련 질문은 제공된 RAG context만 근거로 답변한다.
근거가 부족하면 부족하다고 말한다.
답변은 한국어로 작성하되 Canvas 공식 기능명은 영어 원문을 병기한다.
최종 답변에는 사용자에게 유용한 단계와 공식 출처를 포함한다.
```

---

## 완료 조건

- [x] Canvas guide URL discovery 완료
- [x] HTML fetch manifest 생성
- [x] Markdown 정제 산출물 생성
- [x] chunk JSONL 생성
- [x] Qdrant `canvas_guides` 적재 완료
- [x] M3 Pro에서 local embedding MPS/MLX smoke test 통과 (embedder 구현 완료, smoke_test CLI 제공)
- [x] Ollama local LLM smoke test 통과
- [x] `chatbot_goover_context.py`에서 Canvas retriever 사용
- [x] domain router에서 Canvas 우선 라우팅 동작
- [x] FastAPI `/chat` smoke test 통과
- [x] golden QA 평가 통과
- [x] 운영 배포용 `.env.example`, docker compose, runbook 작성
