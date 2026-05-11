# Claude로 Canvas RAG 챗봇 구축하기

> 이 문서는 Claude Code(CLI)를 사용해 현재 수준의 RAG 챗봇을 처음부터 구축하기 위한
> **최적 프롬프트 전략과 Skills 활용법**을 정리한 가이드입니다.

---

## 목차

1. [Claude Code란?](#1-claude-code란)
2. [프로젝트 전 사전 준비](#2-프로젝트-전-사전-준비)
3. [CLAUDE.md 작성 전략](#3-claudemd-작성-전략)
4. [단계별 최적 프롬프트](#4-단계별-최적-프롬프트)
5. [Claude Skills 활용법](#5-claude-skills-활용법)
6. [Claude Agents 활용법](#6-claude-agents-활용법)
7. [효과적인 프롬프트 작성 원칙](#7-효과적인-프롬프트-작성-원칙)
8. [자주 겪는 문제와 대처법](#8-자주-겪는-문제와-대처법)

---

## 1. Claude Code란?

Claude Code는 Anthropic의 **터미널 기반 AI 코딩 어시스턴트**입니다.

```bash
# 설치
npm install -g @anthropic/claude-code

# 프로젝트 폴더에서 실행
cd my-project
claude
```

일반 ChatGPT와 다른 점:
- **파일을 직접 읽고 쓸 수 있습니다** (Read, Write, Edit 도구)
- **터미널 명령을 실행합니다** (Bash 도구)
- **여러 파일을 동시에 수정합니다**
- **CLAUDE.md를 읽어 프로젝트 규칙을 학습합니다**

---

## 2. 프로젝트 전 사전 준비

### 폴더 구조 먼저 잡기

```bash
mkdir canvas-rag-chatbot
cd canvas-rag-chatbot
git init
```

### 최초 프롬프트 (프로젝트 초기화)

```
canvas-rag-chatbot 프로젝트를 시작합니다.

목표: Instructure Community의 Canvas LMS 공식 가이드를 수집해
한국어 RAG 챗봇을 만드는 것입니다.

개발 환경:
- Apple Silicon M3 Pro (36GB)
- macOS Sonoma
- Docker Desktop 설치됨
- Ollama 설치됨

다음을 해주세요:
1. pyproject.toml (uv 기반) 생성 - fastapi, uvicorn, qdrant-client, httpx 포함
2. 기본 폴더 구조 생성 (src/crawler, src/cleaner, src/indexing, src/retrieval, src/api)
3. .env.example 생성
4. docker-compose.macos.yml 생성 (Qdrant만 먼저)
5. CLAUDE.md 생성 (아래 프로젝트 규칙 기반)

프로젝트 규칙:
- CUDA 코드 없음 (Apple GPU만)
- Canvas 답변은 RAG 컨텍스트 우선, 근거 없으면 추측 금지
- 답변은 한국어, Canvas UI 용어는 영어 병기
- .env는 절대 수정하지 않음 (Hook으로 보호)
```

---

## 3. CLAUDE.md 작성 전략

`CLAUDE.md`는 Claude가 모든 작업 전에 읽는 **프로젝트 헌법**입니다.

### 좋은 CLAUDE.md의 4가지 요소

**① 프로젝트 목적 한 줄 요약**
```markdown
## 프로젝트 목적
Canvas LMS 공식 가이드를 수집해 한국어 RAG 챗봇으로 제공하는 시스템
```

**② 개발 환경 제약 (하드웨어/OS 포함)**
```markdown
## 개발 환경
- Apple Silicon M3 Pro — CUDA 코드 절대 금지
- Qdrant: Docker, LLM: Ollama host, Embedding: MPS 백엔드
```

**③ 절대 원칙 (지켜야 할 규칙)**
```markdown
## 절대 원칙
- Canvas 답변은 RAG 컨텍스트 기반, 근거 없으면 추측 금지
- .env, API Key는 로그에 남기지 않는다
- 테스트 없이 완료로 보고하지 않는다
```

**④ 완료 조건 체크리스트**
```markdown
## 완료 조건
- [x] URL discovery 완료
- [ ] Qdrant 적재 완료
- [ ] FastAPI smoke test 통과
```

### 핵심 팁
> CLAUDE.md가 구체적일수록 Claude가 더 일관된 코드를 작성합니다.
> 특히 "하면 안 되는 것"을 명시하는 것이 중요합니다.

---

## 4. 단계별 최적 프롬프트

### Phase 1: 크롤러 구축

```
src/crawler/ 아래에 Canvas 가이드 크롤러를 구현해주세요.

구현 범위:
1. discover.py — URL 발견
   - 시작: https://community.instructure.com/en/all-guides
   - hostname 제한: community.instructure.com
   - 결과: data/manifests/canvas_urls.jsonl
   - 각 줄: {"url": "...", "title": "...", "role": "...", "crawled_at": "..."}

2. fetch.py — HTML 다운로드
   - 입력: canvas_urls.jsonl
   - 출력: data/raw_html/{hash}.html
   - rate limit: 800ms 간격
   - 이미 받은 URL 건너뛰기 (content_hash로 중복 감지)
   - 실패 시 최대 3회 재시도

3. models.py — Pydantic 데이터 모델

요구사항:
- httpx 비동기 사용
- 동시성 최대 4
- 진행 상황 로그 출력
- 크롤 완료 후 smoke test 명령어도 알려줄 것
```

### Phase 2: 인덱싱 파이프라인

```
src/indexing/ 아래에 Qdrant 인덱싱 파이프라인을 구현해주세요.

구현 범위:
1. chunk.py — 문서 청킹
   - 청크 크기: 900자, 오버랩: 120자
   - 헤딩(##, ###) 기준으로 분할 우선
   - 메타데이터: source_url, title, role, chunk_index, chunk_total, content_hash

2. embedder.py — 임베딩 (우선순위: MPS > MLX > OpenAI)
   - 모델: BAAI/bge-m3
   - MPS 백엔드 (Apple Silicon)
   - fallback 시 로그에 기록

3. qdrant_index.py — Qdrant 적재
   - 컬렉션: canvas_guides
   - payload index: source_url, role, category, product
   - 멱등성: content_hash 같으면 중복 적재 안 함
   - batch upsert (100개씩)

Qdrant가 localhost:6333에서 실행 중이라고 가정합니다.
구현 완료 후 smoke test 명령어를 알려줄 것.
```

### Phase 3: 검색 + 라우터

```
src/retrieval/ 아래에 RAG 검색기와 도메인 라우터를 구현해주세요.

1. retriever.py — CanvasRetriever 클래스
   - search(query, top_k=5, role=None, min_score=0.58)
   - source_url 기준 중복 제거 (같은 문서의 여러 청크 → 최고점 1개)
   - fetch_k = top_k * 3 (더 많이 가져와서 필터링)

2. router.py — DomainRouter 클래스
   - Canvas 키워드(EN + KO) 감지 → canvas 도메인
   - 인사/잡담 감지 → casual 도메인
   - 나머지 → web 도메인
   - has_canvas_history=True이면 짧은 후속 질문도 canvas로 유지

Canvas 키워드 포함:
- EN: module, assignment, quiz, gradebook, rubric, speedgrader, enrollment,
       instructor, student, lti, sis, due date, availability date, submission 등
- KO: 모듈, 과제, 성적, 퀴즈, 루브릭, 강의, 학생, 교수자, 관리자, 마감일 등

구현 후 다음 명령으로 smoke test:
uv run python -m src.retrieval.smoke_test --query "Canvas 모듈 순서 변경"
```

### Phase 4: FastAPI + 스트리밍

```
FastAPI 챗봇 API를 구현해주세요.

파일: src/api/main.py, src/api/chat.py, src/api/models.py

엔드포인트:
- GET  /          → index.html 반환
- GET  /health    → {"status": "ok"}
- POST /chat      → 단순 챗 (ChatRequest → ChatResponse)
- POST /chat/stream → SSE 스트리밍

SSE 이벤트 타입:
- status       : 진행 상태 메시지
- source_found : 검색된 문서 정보
- token        : LLM 토큰 1개
- done         : 완료 + sources + domain

시스템 프롬프트 규칙:
- 답변은 한국어, Canvas UI 용어는 영어 병기
- RAG 컨텍스트 기반으로만 답변
- 근거 없으면 "현재 수집된 문서에서 확인된 근거가 없습니다" 출력
- Canvas 기능 분석 요청(SFR-XXX 형식) → 테이블 형식 답변
- 일반 질문 → 단계별 설명 형식

LLM 설정 (.env 기반):
- LLM_PROVIDER=ollama
- OLLAMA_BASE_URL=http://localhost:11434
- OLLAMA_MODEL=qwen2.5:7b-instruct
```

### Phase 5: 세션 관리 (멀티턴)

```
대화 세션 영속성을 구현해주세요.

1. src/api/session.py — SessionStore 클래스
   - 메모리 캐시: LRU, 최대 200개
   - 세션 없으면 uuid 자동 생성
   - 캐시 미스 시 PostgreSQL에서 히스토리 복원

2. src/api/persistence.py — PostgreSQL CRUD
   - 테이블: chat_sessions (id, title, history JSONB, created_at, updated_at)
   - init_db(), save_session(), load_session(), list_sessions(), delete_session()

3. src/chatbot_goover_context.py — GooverContext 클래스
   - 최대 10턴 히스토리를 LLM 메시지에 포함
   - chat() + stream_chat() 메서드
   - restore_history()로 PostgreSQL 복원

PostgreSQL은 localhost:5433 (Docker)에서 실행 중:
DATABASE_URL=postgresql://chatbot:chatbot@localhost:5433/chatbot

구현 후 서버 재시작 테스트:
1. 질문 → session_id 확인
2. 서버 재시작
3. 같은 session_id로 재질문 → 이전 맥락 유지 확인
```

### Phase 6: ChatGPT 스타일 UI

```
src/api/static/index.html을 ChatGPT 스타일로 완전히 재작성해주세요.

레이아웃:
- 왼쪽 사이드바 (256px, 다크 #202123): 대화 목록
- 오른쪽 메인: 채팅 영역

사이드바:
- "새 대화" 버튼 (+ 아이콘)
- 대화 목록: 오늘/어제/이번 주/이전 그룹별
- 삭제 버튼 (hover 시 표시)
- 하단: 역할 필터 select (학생/교수자/관리자)

도메인 배지:
- canvas: 주황색 (#E66000)
- web: 파란색
- casual: 초록색

소스 표시:
- 스트리밍 중: source-chip으로 문서 제목 표시
- 완료 후: 버블 아래 참고 출처 박스 (오렌지 배경)

기술 요구사항:
- 순수 HTML/CSS/JS (외부 라이브러리 없음)
- SSE 스트리밍 처리 (EventSource 대신 fetch + ReadableStream)
- 마크다운 렌더링 (직접 구현 — marked.js 없이)
- 모바일: 사이드바 오버레이 방식

GET /chat/sessions — 목록
GET /chat/session/{id}/history — 히스토리
DELETE /chat/session/{id} — 삭제
```

### Phase 7: 웹 검색 연동 (SearXNG)

```
자체 호스팅 SearXNG 웹 검색을 연동해주세요.

1. src/retrieval/web_search.py에 SearXNGSearcher 클래스 추가
   - GET {SEARXNG_URL}/search?q=...&format=json
   - 결과: WebSearchResult(title, source_url, content, score)
   - 타임아웃: 10초
   - 실패 시 빈 리스트 반환

2. 라우터 업데이트 (router.py):
   - Canvas 키워드 없음 + 인사 아님 → web 도메인
   - 기존 Tavily 코드는 fallback으로 유지

3. 챗봇 업데이트 (chatbot_goover_context.py):
   - web 도메인: SearXNG 검색 → _WEB_SYSTEM_PROMPT로 LLM 답변
   - canvas 도메인 + RAG 점수 < 0.60: SearXNG 보조 검색 추가

SearXNG Docker 설정을 docker-compose.macos.yml에 추가:
- port: 8888
- SEARXNG_URL=http://localhost:8888 (.env에 추가)
```

---

## 5. Claude Skills 활용법

Skills는 `.claude/skills/` 폴더에 저장된 **재사용 가능한 작업 지침**입니다.

### 현재 프로젝트의 Skills

```
.claude/skills/
├── canvas-rag-crawl/SKILL.md     # 크롤러 실행 절차
├── canvas-rag-ingest/SKILL.md    # 인덱싱 파이프라인 실행
├── canvas-rag-chatbot/SKILL.md   # 챗봇 서버 실행
└── canvas-rag-eval/SKILL.md      # 평가 실행
```

### `/canvas-rag-crawl` Skill 예시

```markdown
# Canvas RAG 크롤링 Skill

이 Skill을 실행하면 Canvas 공식 가이드를 크롤링합니다.

## 실행 전 확인
1. Docker Desktop 실행 중인지 확인
2. data/manifests/ 폴더 존재 확인

## 실행 순서
1. URL 발견:
   uv run python -m src.crawler.discover \
     --start-url "https://community.instructure.com/en/all-guides" \
     --out data/manifests/canvas_urls.jsonl

2. HTML 다운로드:
   uv run python -m src.crawler.fetch \
     --manifest data/manifests/canvas_urls.jsonl \
     --out-dir data/raw_html --concurrency 4 --delay-ms 800

3. 완료 확인:
   ls data/raw_html/ | wc -l  # 다운로드된 파일 수

## 오류 처리
- 네트워크 오류: --delay-ms를 1200으로 늘려 재시도
- 이미 완료: --skip-existing 플래그로 건너뜀
```

### Skill 호출 방법

```
# Claude에게 Skill 실행 요청
/canvas-rag-crawl

# 또는 자연어로
"크롤링 Skill 실행해줘"
```

### Skill 작성 팁

- **전제 조건** 명시 (어떤 서비스가 실행 중이어야 하는지)
- **실행 순서** 번호로 나열
- **검증 명령어** 포함 (실행 후 확인 방법)
- **오류 처리** 패턴 포함

---

## 6. Claude Agents 활용법

Agents는 `.claude/agents/` 폴더에 저장된 **전문 역할 AI**입니다.

### 현재 프로젝트의 Agents

```
.claude/agents/
├── crawler-engineer.md          # 크롤러 전문가
├── rag-indexing-engineer.md     # 임베딩/Qdrant 전문가
├── chatbot-api-engineer.md      # FastAPI/LLM 전문가
├── rag-qa-evaluator.md          # 평가 전문가
└── macos-mlx-runtime-engineer.md # Apple Silicon 최적화 전문가
```

### Agent 활용 예시

복잡한 작업에서 적합한 Agent를 명시:

```
[crawler-engineer 에이전트에게]
현재 discover.py에서 Canvas 카테고리 페이지가 누락되고 있습니다.
/en/kb/collections/{slug} 패턴의 URL도 수집해야 합니다.
discover.py를 수정해주세요.
```

```
[macos-mlx-runtime-engineer 에이전트에게]
embedder.py에서 MPS 백엔드 사용 시 메모리 오류가 납니다.
error: "MPS backend out of memory"
bge-m3 모델로 1000개 청크를 배치 임베딩할 때 발생합니다.
배치 크기 조정 및 메모리 관리 방법을 구현해주세요.
```

### Agent 정의 파일 예시 (`chatbot-api-engineer.md`)

```markdown
# Chatbot API Engineer

## 역할
FastAPI 챗봇 서버, SSE 스트리밍, LLM 연동, 세션 관리를 담당합니다.

## 전문 영역
- FastAPI 엔드포인트 설계
- SSE(Server-Sent Events) 스트리밍 구현
- Ollama/OpenAI 클라이언트 연동
- GooverContext 멀티턴 세션 관리
- 시스템 프롬프트 엔지니어링

## 사용 파일
- src/api/main.py
- src/api/chat.py
- src/api/models.py
- src/api/session.py
- src/chatbot_goover_context.py

## 금지 사항
- .env 파일 직접 수정
- API Key 로그 출력
- blocking I/O in async context
```

---

## 7. 효과적인 프롬프트 작성 원칙

### 원칙 1: 범위를 명확히 하라

나쁜 예:
```
챗봇을 만들어줘
```

좋은 예:
```
src/api/chat.py의 _SYSTEM_PROMPT에 Canvas 역할별 권한 규칙을 추가해주세요.
학생(Student)은 모듈 순서 변경 불가, 과제 생성/편집 불가를 명시해야 합니다.
다른 파일은 수정하지 말고 chat.py만 수정해주세요.
```

### 원칙 2: 파일 경로와 라인 번호를 제공하라

```
src/api/static/index.html의 691번째 줄에서
sourcesHtml이 .bubble 안에 들어가 있어서 참고 출처가 보이지 않습니다.
sourcesHtml을 .bubble 밖으로 이동시켜 주세요.
```

### 원칙 3: 왜(Why)를 설명하라

```
bge-m3의 한국어→영어 교차 언어 검색에서 관련 문서가
0.58~0.65 구간에 집중됩니다. 그래서 min_score를 0.63으로 올리면
노이즈는 줄지만 일부 관련 문서가 누락될 수 있습니다.
이 트레이드오프를 고려해서 임계값 조정 방법을 알려주세요.
```

### 원칙 4: 제약 조건을 먼저 말하라

```
다음 제약 조건 하에서 구현해주세요:
- CUDA 코드 없음 (Apple Silicon만)
- 외부 라이브러리 추가 없음 (현재 pyproject.toml 유지)
- .env 파일 수정 없음
- 기존 API 인터페이스 변경 없음
```

### 원칙 5: 검증 방법을 요청하라

```
구현 완료 후 다음을 확인해주세요:
1. 서버 재시작
2. "Canvas 모듈 순서 변경" 질문으로 스트리밍 테스트
3. 참고 출처가 답변 아래에 표시되는지 확인
```

---

## 8. 자주 겪는 문제와 대처법

### 문제 1: Claude가 .env를 직접 수정하려 한다

**원인**: .env 파일에 민감한 정보가 있어 Claude 도구로 수정을 막아야 함

**해결**: `.claude/hooks/protect-files.sh` Hook 설정

```bash
#!/bin/bash
# .env 파일 직접 편집 차단
if echo "$CLAUDE_TOOL_INPUT" | grep -q '".env"'; then
  echo "ERROR: .env 파일은 직접 수정할 수 없습니다. 수동으로 편집하세요."
  exit 1
fi
```

### 문제 2: Claude가 너무 많은 파일을 동시에 수정한다

**대처**: 범위를 제한하는 지시 추가

```
"이번 수정은 src/api/chat.py 파일만 수정합니다.
다른 파일은 변경하지 마세요."
```

### 문제 3: 이전 대화 내용을 잊어버린다 (컨텍스트 소진)

**대처**: CLAUDE.md에 현재 상태 기록 + `/compact` 명령 사용

```
# 현재 개발 상태를 CLAUDE.md 하단에 추가해줘
## 현재 완료된 작업
- [x] 크롤러 구현
- [x] Qdrant 적재
- [ ] FastAPI 구현 (진행 중)
```

### 문제 4: 테스트 없이 "완료"라고 보고한다

**대처**: CLAUDE.md에 명시

```markdown
## 절대 원칙
- 테스트 없이 완료로 보고하지 않는다
- 모든 구현은 smoke test 명령어와 예상 출력을 함께 제공한다
```

### 문제 5: Apple Silicon에서 CUDA 코드를 작성한다

**대처**: CLAUDE.md에 명시 + 프롬프트마다 강조

```markdown
## 개발 환경 제약
- CUDA 전제 코드 절대 금지
- GPU 사용: MPS(PyTorch), MLX(Apple), Ollama(Metal) 중 하나
- torch.cuda.is_available() 대신 torch.backends.mps.is_available() 사용
```

### 문제 6: 한국어 요청인데 영어로 코드 주석을 단다

**대처**: 명시적 지시

```
코드 주석은 영어로 작성하되,
사용자 메시지와 오류 메시지는 한국어로 작성해주세요.
```

---

## 빠른 참조: 프롬프트 템플릿

### 버그 수정 템플릿

```
[파일명:라인번호]에서 [증상]이 발생합니다.
원인: [원인 설명]
수정 범위: [파일명]만 수정
수정 후 [검증 방법]으로 확인해주세요.
```

### 새 기능 추가 템플릿

```
[파일명]에 [기능명]을 추가해주세요.

요구사항:
- [요구사항 1]
- [요구사항 2]

제약:
- 기존 [인터페이스/API] 변경 없음
- [금지 사항]

완료 기준:
- [검증 명령어]로 [예상 결과] 확인
```

### 리팩토링 템플릿

```
[파일명]의 [함수/클래스]를 리팩토링해주세요.

현재 문제: [문제 설명]
원하는 상태: [목표 설명]

변경하면 안 되는 것:
- 외부 인터페이스 (함수 시그니처, API 경로)
- 테스트 파일
```
