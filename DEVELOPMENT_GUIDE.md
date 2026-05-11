# Canvas RAG 챗봇 — 개발 가이드

> 이 문서는 프로젝트의 전체 구조와 동작 원리를 **처음 접하는 분**도 이해할 수 있도록 작성되었습니다.

---

## 목차

1. [이 프로젝트가 무엇인가요?](#1-이-프로젝트가-무엇인가요)
2. [핵심 개념 설명](#2-핵심-개념-설명)
3. [전체 시스템 구조](#3-전체-시스템-구조)
4. [기술 스택](#4-기술-스택)
5. [각 구성 요소 상세 설명](#5-각-구성-요소-상세-설명)
6. [데이터 흐름 (질문 → 답변)](#6-데이터-흐름-질문--답변)
7. [API 엔드포인트 정리](#7-api-엔드포인트-정리)
8. [환경 설정 및 실행 방법](#8-환경-설정-및-실행-방법)
9. [주요 설정 파일 설명](#9-주요-설정-파일-설명)

---

## 1. 이 프로젝트가 무엇인가요?

Canvas LMS(학습관리시스템)의 **공식 가이드 문서를 AI가 읽고**, 사용자 질문에 **한국어로 근거 있는 답변**을 제공하는 챗봇입니다.

### 예시

```
사용자: "Canvas에서 due date와 availability date 차이가 뭔가요?"

AI: Canvas LMS에서 두 날짜는 다음과 같이 구분됩니다.

- Due Date (마감일): 과제를 제출해야 하는 날짜입니다. 이 날짜 이후 제출은
  '지각(Late)'으로 처리됩니다.
- Availability Date (이용 가능 날짜): 과제 자체에 접근할 수 있는 기간입니다.
  Available From ~ Until 범위를 벗어나면 과제 페이지 자체가 잠깁니다.

참고 출처: [Assignments Overview - Canvas Guide] (https://...)
```

단순히 ChatGPT에게 물어보는 것과의 차이점은 **공식 문서를 근거로 답변**하기 때문에 잘못된 정보(할루시네이션)가 적습니다.

---

## 2. 핵심 개념 설명

### RAG (Retrieval-Augmented Generation) — 검색 증강 생성

AI가 답변할 때 **문서를 먼저 검색한 뒤** 그 내용을 바탕으로 답변을 생성하는 방식입니다.

```
일반 LLM:  질문 → AI 기억 → 답변 (학습된 지식 기반, 오래되거나 틀릴 수 있음)
RAG:       질문 → 문서 검색 → 관련 문서 + 질문 → AI → 답변 (최신 문서 기반)
```

### 벡터 검색 (Vector Search)

텍스트를 **숫자 배열(벡터)** 로 변환해서 의미적으로 유사한 문서를 찾는 기술입니다.

```
"과제 제출 방법" → [0.12, -0.34, 0.87, ...] (1024차원 숫자)
"assignment submission" → [0.11, -0.31, 0.85, ...] (비슷한 숫자 → 유사!)
```

키워드가 달라도 **의미가 비슷하면** 찾아냅니다. 한국어로 물어봐도 영어 문서를 찾을 수 있는 이유입니다.

### SSE (Server-Sent Events) — 실시간 스트리밍

AI 답변이 생성되는 대로 **글자 단위로 화면에 표시**되는 기술입니다. ChatGPT처럼 텍스트가 흘러나오는 효과입니다.

### 세션 (Session)

대화의 **기억 단위**입니다. 같은 세션 안에서는 이전 대화 내용을 기억하고 맥락을 이어갑니다.

---

## 3. 전체 시스템 구조

### 데이터 수집 파이프라인 (1회성 작업)

```
Instructure Community (공식 Canvas 가이드 웹사이트)
    │
    ▼
[크롤러 - discover.py]       URL 목록 수집
    │
    ▼
[크롤러 - fetch.py]          HTML 파일 다운로드
    │
    ▼
[정제기 - to_markdown.py]    HTML → Markdown 변환 (불필요한 태그 제거)
    │
    ▼
[청커 - chunk.py]            긴 문서를 900자 단위 조각으로 분할
    │
    ▼
[임베더 - embedder.py]       텍스트 → 숫자 벡터 변환 (bge-m3 모델)
    │
    ▼
[Qdrant DB]                  벡터 데이터베이스에 저장 (8,408개 청크)
```

### 실시간 챗봇 파이프라인 (매 질문마다)

```
사용자 질문
    │
    ▼
[도메인 라우터 - router.py]
    ├── Canvas 키워드? → Canvas RAG 경로
    ├── 웹 검색 필요? → SearXNG 경로
    └── 인사/잡담?    → 직접 LLM 경로
    │
    ▼ (Canvas RAG 경로)
[검색기 - retriever.py]      질문을 벡터로 변환 → Qdrant에서 유사 문서 검색
    │
    ▼
[LLM - Ollama/qwen2.5]       문서 + 질문 → 한국어 답변 생성
    │
    ▼
[FastAPI - main.py]          HTTP 응답 반환 (SSE 스트리밍)
    │
    ▼
[웹 UI - index.html]         ChatGPT 스타일 채팅 화면 표시
```

### 저장소 구조

```
canvas-rag-chatbot/
├── src/
│   ├── crawler/                # 웹 크롤러
│   │   ├── discover.py         # URL 발견
│   │   └── fetch.py            # HTML 다운로드
│   ├── cleaner/
│   │   └── to_markdown.py      # HTML → Markdown 정제
│   ├── indexing/
│   │   ├── chunk.py            # 청킹 (문서 분할)
│   │   ├── embedder.py         # 텍스트 → 벡터 변환
│   │   └── qdrant_index.py     # Qdrant 적재
│   ├── retrieval/
│   │   ├── retriever.py        # 벡터 검색
│   │   ├── router.py           # 도메인 라우팅
│   │   └── web_search.py       # SearXNG 웹 검색
│   ├── api/
│   │   ├── main.py             # FastAPI 서버
│   │   ├── chat.py             # 챗봇 로직 + 시스템 프롬프트
│   │   ├── models.py           # 요청/응답 데이터 형식
│   │   ├── session.py          # 세션 관리 (메모리 캐시)
│   │   ├── persistence.py      # PostgreSQL 저장
│   │   └── static/index.html   # 웹 채팅 UI
│   └── chatbot_goover_context.py  # 대화 세션 컨텍스트 관리
├── data/                       # 크롤링/처리 데이터 (gitignore)
├── .env                        # 환경변수 (비공개)
├── .env.example                # 환경변수 예시
├── docker-compose.macos.yml    # Qdrant + SearXNG + PostgreSQL 컨테이너
└── pyproject.toml              # Python 의존성
```

---

## 4. 기술 스택

| 역할 | 도구 | 설명 |
|------|------|------|
| **웹 프레임워크** | FastAPI | Python 기반 고성능 API 서버 |
| **LLM (언어 모델)** | Ollama + qwen2.5:7b | 로컬에서 실행되는 AI (Apple GPU 사용) |
| **임베딩 모델** | bge-m3 (BAAI) | 텍스트를 벡터로 변환 (한/영 모두 지원) |
| **벡터 데이터베이스** | Qdrant | 벡터 검색 전용 DB |
| **관계형 DB** | PostgreSQL | 대화 세션 영구 저장 |
| **웹 검색** | SearXNG | 자체 호스팅 검색 엔진 |
| **크롤러** | httpx + BeautifulSoup | Canvas 공식 문서 수집 |
| **패키지 관리** | uv | 빠른 Python 패키지 설치 |
| **컨테이너** | Docker | Qdrant/SearXNG/PostgreSQL 실행 |
| **UI** | 순수 HTML/CSS/JS | ChatGPT 스타일 다크 사이드바 인터페이스 |

### 왜 로컬 AI인가?

- **비용**: OpenAI API 비용 없음
- **보안**: 사내 질문이 외부 서버로 전송되지 않음
- **속도**: Apple M3 Pro Metal GPU로 가속
- **오프라인**: 인터넷 없이도 동작

---

## 5. 각 구성 요소 상세 설명

### 5-1. 도메인 라우터 (`src/retrieval/router.py`)

사용자 질문이 어떤 종류인지 자동으로 판별합니다.

| 도메인 | 조건 | 처리 방식 |
|--------|------|-----------|
| `canvas` | "모듈", "과제", "assignment" 등 Canvas 키워드 포함 | Qdrant RAG 검색 |
| `web` | Canvas와 무관한 일반 질문 | SearXNG 웹 검색 |
| `casual` | 인사말, 잡담 ("안녕", "고마워") | LLM 직접 답변 |
| `internal` | 사내 정책 문서 (미래 확장용) | 별도 RAG |

**Canvas 후속 질문 처리**: 이전 대화가 Canvas 주제였다면, 짧은 참조 표현("그건?", "이 경우는?")도 자동으로 Canvas로 라우팅합니다.

### 5-2. 검색기 (`src/retrieval/retriever.py`)

```
1. 질문 텍스트 → bge-m3 모델로 벡터(숫자 배열) 변환
2. Qdrant에서 코사인 유사도 기준 상위 45개 후보 검색
3. min_score=0.58 미만 필터링 (관련 없는 문서 제거)
4. 같은 URL 중복 제거 (같은 페이지의 여러 청크 → 최고점 1개만)
5. 상위 top_k(기본 15)개 반환
```

### 5-3. 챗봇 컨텍스트 (`src/chatbot_goover_context.py`)

대화 세션 하나를 관리하는 핵심 클래스입니다.

- **히스토리 관리**: 최대 10턴의 이전 대화를 LLM에 전달
- **3가지 답변 경로**: casual(직접) / web(SearXNG) / canvas(RAG)
- **스트리밍**: SSE로 토큰 단위 실시간 전달

### 5-4. 시스템 프롬프트 (`src/api/chat.py`)

LLM에게 전달하는 **행동 지침**입니다. 주요 규칙:

- 답변은 반드시 한국어
- Canvas UI 기능명은 영어 원문 병기
- RAG 근거 없으면 추측 금지
- CASE A(기능 분석 테이블) vs CASE B(일반 답변) 자동 판단
- **학생(Student) 권한 제약**: 모듈 순서 변경 불가 등 역할별 권한 명시

### 5-5. 세션 저장 (`src/api/session.py` + `src/api/persistence.py`)

```
사용자가 질문
    ↓
메모리 캐시(LRU, 최대 200개) 확인
    ├── 있음: 캐시에서 즉시 로드
    └── 없음: PostgreSQL에서 히스토리 복원
        ↓
대화 완료 후 PostgreSQL에 자동 저장
    ↓
서버 재시작 후에도 이전 대화 유지
```

### 5-6. 웹 UI (`src/api/static/index.html`)

ChatGPT와 유사한 인터페이스:

- **왼쪽 사이드바**: 오늘/어제/이번 주별 대화 목록
- **새 대화 버튼**: 세션 초기화
- **도메인 배지**: Canvas RAG(주황), 웹 검색(파랑), 일상 대화(초록)
- **참고 출처**: 답변 아래 링크로 표시
- **실시간 스트리밍**: 토큰 단위로 텍스트 출력

---

## 6. 데이터 흐름 (질문 → 답변)

Canvas 질문 "모듈 순서를 바꾸는 방법은?" 예시:

```
① 사용자가 질문 입력 → POST /chat/stream 요청

② 서버: 라우터가 "모듈" 키워드 감지 → domain="canvas"

③ 서버: SSE 이벤트 전송
   data: {"type": "status", "message": "Canvas 공식 문서에서 검색 중..."}

④ 서버: Qdrant에서 "모듈 순서 변경" 관련 청크 15개 검색
   data: {"type": "source_found", "title": "How do I reorder modules...", "score": 0.71}
   ... (15번 반복)

⑤ 서버: LLM에게 [시스템 프롬프트 + 이전 대화 + 15개 문서 + 질문] 전달

⑥ LLM이 답변 생성 → 토큰 단위 스트리밍
   data: {"type": "token", "content": "Canvas"}
   data: {"type": "token", "content": " LMS에서"}
   ...

⑦ 답변 완료
   data: {"type": "done", "domain": "canvas", "sources": [...], "session_id": "..."}

⑧ 브라우저: 답변 렌더링 + 참고 출처 표시 + 사이드바 갱신
```

---

## 7. API 엔드포인트 정리

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 웹 채팅 UI 반환 |
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/chat` | 단순 채팅 (스트리밍 없음) |
| `POST` | `/chat/stream` | **실시간 SSE 스트리밍 채팅** (UI가 사용) |
| `GET` | `/chat/sessions` | 전체 대화 목록 조회 |
| `GET` | `/chat/session/{id}/history` | 특정 대화의 전체 히스토리 |
| `DELETE` | `/chat/session/{id}` | 대화 삭제 |
| `POST` | `/chat/session/{id}/reset` | 대화 히스토리 초기화 (ID 유지) |

### 요청 예시 (`/chat/stream`)

```json
{
  "query": "Canvas에서 모듈 순서를 바꾸는 방법은?",
  "role": "instructor",
  "session_id": "abc123",
  "top_k": 15
}
```

---

## 8. 환경 설정 및 실행 방법

### 사전 요구사항

- macOS (Apple Silicon M1/M2/M3 권장)
- Docker Desktop 실행 중
- Ollama 설치 (`brew install ollama`)
- uv 설치 (`brew install uv`)

### 1단계: 환경변수 설정

`.env.example`을 복사해 `.env` 생성:

```bash
cp .env.example .env
```

`.env` 내용:

```
DATABASE_URL=postgresql://chatbot:chatbot@localhost:5433/chatbot
SEARXNG_URL=http://localhost:8888
EMBEDDING_PROVIDER=local_mps
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=canvas_guides
```

### 2단계: 인프라 실행 (Docker)

```bash
docker compose -f docker-compose.macos.yml up -d
```

Qdrant(6333), SearXNG(8888), PostgreSQL(5433) 컨테이너가 시작됩니다.

### 3단계: LLM 모델 준비

```bash
ollama serve               # Ollama 서버 시작
ollama pull qwen2.5:7b-instruct  # AI 모델 다운로드 (약 4.7GB)
```

### 4단계: Python 의존성 설치

```bash
uv sync                    # 기본 의존성
uv sync --group embedding  # 임베딩 모델 추가 (sentence-transformers)
```

### 5단계: 서버 실행

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

브라우저에서 `http://localhost:8080` 접속

### 데이터 수집 (최초 1회)

Canvas 공식 문서가 이미 수집되어 있다면 건너뜁니다.

```bash
# URL 수집
uv run python -m src.crawler.discover \
  --start-url "https://community.instructure.com/en/all-guides" \
  --out data/manifests/canvas_urls.jsonl

# HTML 다운로드
uv run python -m src.crawler.fetch \
  --manifest data/manifests/canvas_urls.jsonl \
  --out-dir data/raw_html

# Markdown 변환
uv run python -m src.cleaner.to_markdown \
  --input-manifest data/manifests/canvas_urls.jsonl \
  --raw-html-dir data/raw_html \
  --out-dir data/markdown \
  --out-manifest data/manifests/canvas_docs.jsonl

# 청킹
uv run python -m src.indexing.chunk \
  --manifest data/manifests/canvas_docs.jsonl \
  --out data/chunks/canvas_chunks.jsonl

# Qdrant 적재
uv run python -m src.indexing.qdrant_index \
  --chunks data/chunks/canvas_chunks.jsonl \
  --collection canvas_guides
```

---

## 9. 주요 설정 파일 설명

### `CLAUDE.md`

Claude AI에게 이 프로젝트의 규칙과 아키텍처를 설명하는 파일입니다. Claude가 코드를 작성할 때 이 파일을 읽고 프로젝트 컨벤션을 따릅니다.

### `pyproject.toml`

Python 의존성 목록입니다. `uv sync`로 설치합니다.

### `docker-compose.macos.yml`

개발 환경용 컨테이너 설정입니다. Qdrant, SearXNG, PostgreSQL을 한 번에 실행합니다.

### `.claude/settings.json`

Claude Code 설정 및 Git Hook 설정입니다. `.env` 파일 직접 수정을 방지하는 보호 훅이 포함됩니다.
