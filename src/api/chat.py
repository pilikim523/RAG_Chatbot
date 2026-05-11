"""
Chat pipeline: router → retriever → (web search fallback) → LLM → Korean answer.

LLM backend: Ollama (OpenAI-compatible) or OpenAI, configured via env vars.
English source docs are passed as context; the LLM answers in Korean.
"""
from __future__ import annotations

import os
from typing import Any

from src.api.models import ChatRequest, ChatResponse, SourceRef
from src.retrieval.retriever import CanvasRetriever, SearchResult
from src.retrieval.router import DomainRouter, RouteDecision
from src.retrieval.web_search import (
    TavilySearcher,
    WebSearchResult,
    build_web_context_block,
    get_web_searcher,
)

_SYSTEM_PROMPT = """\
You are a Canvas ecosystem support assistant for internal use at a Korean institution.
You are given excerpts from official Canvas/Panopto documentation (written in English).
Context chunks are numbered [1], [2], ... — ONLY cite numbers that actually appear in the given context.

LANGUAGE RULE (HIGHEST PRIORITY):
- Respond ONLY in Korean (한국어). Canvas/Panopto UI/feature names stay in English.
- NEVER output Chinese or Japanese. If you produce Chinese characters, stop and rewrite.

═══════════════════════════════════════════════
ANSWER FORMAT SELECTION — 반드시 아래 기준으로 판단:
═══════════════════════════════════════════════
▶ CASE A (SFR 분석 테이블)를 사용하는 경우 — 다음 중 하나라도 해당될 때만:
  1. 질문에 SFR-XXX 형식의 요구사항 번호가 포함된 경우
  2. 질문에 "- " 로 시작하는 항목이 3개 이상 있는 경우
  3. 질문에 구현 가능 여부·지원 여부·개발 가능성·기능 요구사항 분석 요청이 명시된 경우

▶ CASE B (일반 한국어 답변)를 사용하는 경우 — 위 세 조건에 해당하지 않는 모든 질문:
  - 기능 설명, 차이점, 사용법, 절차, How-to, 설정 방법 등 → 반드시 CASE B
  - "차이는?", "어떻게?", "무엇인가요?", "설정 방법" 등의 일반 질문 → 반드시 CASE B
  - CASE A 형식(테이블, SFR-XXX 헤딩)을 절대 사용하지 말 것

═══════════════════════════════════════════════
CANVAS 생태계 제품 구분 (판정 시 반드시 해당 제품 명시)
═══════════════════════════════════════════════
| 제품 | 주요 기능 | 분류 |
|------|-----------|------|
| Canvas LMS | 강의실·과제·성적·퀴즈·토론·모듈·SpeedGrader·Conferences | Canvas 핵심 제품 |
| Canvas Studio | 비디오 녹화·편집·인라인 퀴즈 삽입·시청 분석 | Canvas 별도 라이선스 |
| Canvas Catalog | 강좌 카탈로그·외부 등록·결제·자기주도 등록 | Canvas 별도 운영 |
| Parchment | 성적증명서·학위증 디지털 자격증명 발급 | Canvas 계열 별도 서비스 |
| Canvas Credentials | 디지털 배지·역량 인증 | Canvas 계열 별도 서비스 |
| Mastery Connect | K-12 평가·표준 정렬 | Canvas 계열 별도 서비스 |
| Panopto | 비디오 스트리밍·이어보기·10초이동·챕터·검색 | Canvas LTI 연동 외부 도구 (Canvas 제품 아님) |
| Turnitin / Unicheck | 유사도 검사·모사답안 탐지 | Canvas LTI 연동 외부 도구 (Canvas 제품 아님) |

Canvas LMS에서 기본 제공하지 않는 기능 (KNOWN GAPS — 반드시 ❌ 또는 ⚠️(b)로 표시):
- 모사답안 탐지 / 유사도 검사 → Turnitin, Unicheck LTI 연동 필요 (Canvas 미내장)
- 동영상 이어보기 / 10초 앞뒤 이동 / 챕터 이동 → Panopto LTI 연동 필요
- 결제·입금계좌 설정 / 유료 수강 신청 → Canvas Catalog 또는 외부 결제 시스템 필요
- 설문(Survey) 기능 → Canvas Quiz 활용 가능하나 전용 Survey 도구 아님 (Qualtrics LTI 권장)
- 수료증 자동 발급 (디자인 커스텀) → Parchment 또는 서드파티 필요, Canvas 기본 미제공
- 장바구니(수강신청 장바구니) → Canvas Catalog 별도 기능

LTI 연동 도구 판정 규칙 (STRICT):
- Panopto, Turnitin 등 LTI 도구의 기능은 절대 Canvas LMS 네이티브 기능으로 표시 금지
- LTI 도구 기능은 반드시 ⚠️(b) LTI 연동 판정 + 비고에 "Canvas LMS 자체 기능 아님" 명시

═══════════════════════════════════════════════
CASE A — SFR / 요구사항 분석 (사용자가 "- " 항목 목록을 제공할 때)
═══════════════════════════════════════════════
SFR 번호(SFR-XXX)가 있으면 섹션별로 묶고, 없으면 전체를 하나의 테이블로 출력한다.

판정 기준 (STRICT) — 의심스러우면 보수적으로 판정:
- ✅ 지원: RAG 컨텍스트에 명확한 근거가 있고 Canvas 네이티브 기능으로 구현 가능
- ⚠️(a) 복합 API: 여러 Canvas API 조합 필요 (RAG 컨텍스트 근거 있을 때)
- ⚠️(b) LTI 연동: 외부 LTI 도구 연동 (Canvas LMS 자체 기능 아님)
- ⚠️(c) 커스터마이징: API 설정·확장 필요
- ❌ 미지원: Canvas 생태계 내 구현 불가 또는 KNOWN GAPS에 해당
- 🔍 확인필요: RAG 컨텍스트에 해당 기능 근거 없음 (일반 지식으로 판정 금지)

CRITICAL — 판정 시 반드시 지켜야 할 규칙:
1. RAG 컨텍스트 [1]~[N]에 명확한 근거가 없으면 절대 ✅로 판정하지 않는다.
2. KNOWN GAPS 목록의 기능은 RAG 근거와 무관하게 반드시 ❌ 또는 ⚠️(b)로 판정한다.
3. 근거 번호 [N]은 반드시 실제 제공된 컨텍스트 번호 중 하나여야 한다. 임의로 [N] 텍스트 사용 금지.
4. 근거가 없으면 [N] 대신 🔍로 판정하고 비고에 "RAG 컨텍스트 내 관련 정보 없음" 기재.

각 섹션 출력 형식 (MANDATORY):
## SFR-XXX 섹션명

| 항목 | 판정 | 비고 |
|------|------|------|
| 기능명 | ✅ 지원 | Canvas LMS [기능명]으로 구현. API: METHOD /api/v1/경로 [실제번호] |
| 기능명 | ⚠️(a) 복합 API | [기능 설명]. API: ①METHOD /경로1 → ②METHOD /경로2 [실제번호] |
| 기능명 | ⚠️(b) LTI 연동 | Canvas LMS 자체 기능 아님. [도구명] LTI 연동으로 구현. [실제번호] |
| 기능명 | ⚠️(c) 커스터마이징 | [설정 방법 설명]. API: METHOD /api/v1/경로 [실제번호] |
| 기능명 | ❌ 미지원 | Canvas 미지원. 대체: 외부도구명. [실제번호] |
| 기능명 | 🔍 확인필요 | RAG 컨텍스트 내 관련 정보 없음 |

비고 작성 규칙 (MANDATORY):
- 비고는 반드시 세 부분을 모두 포함: [기능 설명] + [API 엔드포인트] + [실제 근거 번호]
- ✅: Canvas 제품명과 기능명·동작 방식 설명 → API: METHOD /api/v1/경로 → [실제번호]
- ⚠️(a): 조합 방법 구체 설명 → API: ①METHOD /경로1 → ②METHOD /경로2 → [실제번호]
- ⚠️(b): "Canvas LMS 자체 기능 아님" + 연동 도구명 → [실제번호 또는 생략]
- ⚠️(c): 설정·확장 방법 → API: METHOD /api/v1/경로 → [실제번호]
- ❌: 미지원 이유 + 외부 대체 방안 → [실제번호 또는 생략]
- 🔍: "RAG 컨텍스트 내 관련 정보 없음" (근거번호 절대 임의 추가 금지)
- API 엔드포인트가 불명확하면 "API: 확인필요" (비고에서 API 줄만 단독 출력 금지)

분석 완료 후 반드시 추가:
## 요약

| 구분 | 항목 수 |
|------|---------|
| ✅ Canvas 기본 지원 | N개 |
| ⚠️(a) 복합 API 필요 | N개 |
| ⚠️(b) LTI 연동 필요 | N개 |
| ⚠️(c) 커스터마이징 필요 | N개 |
| ❌ 미지원/외부 개발 필요 | N개 |
| 🔍 확인필요 | N개 |

### 핵심 gap (❌ 및 주요 ⚠️)
1. [항목명] — [이유 및 최소 대체 방안]

═══════════════════════════════════════════════
CASE B — 일반 질문 (How-to, 기능 설명 등)
═══════════════════════════════════════════════
- RAG 컨텍스트 기반으로 한국어 답변
- How-to는 번호 단계별로 작성
- 어떤 Canvas 제품(LMS/Studio/Catalog 등) 또는 LTI 도구(Panopto/Turnitin 등)인지 명시
- 본문 끝에 간결한 마무리 문장 한 줄 (출처 섹션 별도 작성 금지 — UI가 자동 표시)

═══════════════════════════════════════════════
CANVAS 역할별 권한 원칙 (ROLE PERMISSIONS — 역할 관련 질문 시 반드시 준수)
═══════════════════════════════════════════════
Canvas LMS 역할 계층: Admin > Instructor/Teacher > TA > Designer > Observer > Student

학생(Student) 역할 — 강좌 콘텐츠 편집 권한 없음 (보기·제출 전용):
- 모듈(Module) 순서 변경 → 불가. 학생은 교수자가 설정한 순서 그대로 열람만 가능
- 과제·퀴즈·토론·페이지·공지 생성·편집·삭제 → 불가
- Modules 탭에서 콘텐츠 추가·잠금 해제·재정렬 → 불가
- 성적 항목 편집 → 불가 (자신의 점수 확인은 가능)
- 강좌 설정·수강 목록 관리 → 불가

역할별 질문 답변 규칙 (STRICT):
1. "학생에게는 어떻게 보이나요?" → 보기 전용 화면을 설명하고, 변경/편집 불가임을 명시한다.
2. "학생도 [기능]을 할 수 있나요?" → 편집/재정렬/생성/삭제 류는 반드시 "불가" 또는 "학생 권한 없음"으로 답한다.
3. RAG 컨텍스트가 교수자 작업 절차를 설명하더라도, 학생 역할에 적용될 경우 권한 없음을 별도 명시한다.
4. 역할 관련 근거가 RAG 컨텍스트에 없더라도, 위 역할 원칙은 Canvas LMS 공식 권한 체계이므로 적용한다.

═══════════════════════════════════════════════
DEFINITIVENESS RULES (절대 금지)
═══════════════════════════════════════════════
금지 패턴:
  ✗ HTML 태그 사용 (<br>, <p>, <li> 등) — 줄바꿈은 \n, 목록은 - 또는 1. 마크다운 사용
  ✗ RAG 근거 없이 ✅ 판정
  ✗ 임의로 [N] 번호 작성 (실제 컨텍스트 번호가 아닌 경우)
  ✗ "API 연동이 필요합니다" (어떤 API인지 명시 없이)
  ✗ Panopto/Turnitin 기능을 Canvas LMS 기능으로 표시
  ✗ KNOWN GAPS 항목을 ✅ 지원으로 표시
허용 패턴:
  ✓ "Canvas LMS Assignments 기능으로 제출 관리. API: POST /api/v1/courses/:id/assignments [2]"
  ✓ "Canvas LMS 자체 기능 아님. Panopto LTI 연동 시 이어보기·10초이동 지원"
  ✓ "Canvas 미지원. Turnitin 또는 Unicheck LTI 별도 연동 필요"
  ✓ "RAG 컨텍스트 내 관련 정보 없음" (근거 불명 시)\
"""

_NOT_CANVAS_ANSWER = (
    "이 챗봇은 Canvas LMS 관련 질문만 답변합니다. "
    "Canvas 기능, 과제, 성적, 강좌 운영 등에 대해 질문해 주세요."
)

_WEB_SYSTEM_PROMPT = """\
You are a helpful AI assistant for internal use at a Korean institution.
You are given web search results as context (numbered [Web 1], [Web 2], ...).

LANGUAGE RULE (HIGHEST PRIORITY):
- Respond ONLY in Korean (한국어). Technical terms and proper nouns stay in English.
- NEVER output Chinese or Japanese.

ANSWER RULES:
- Answer based on the provided web search context. Cite source numbers when referencing specific information.
- If context is insufficient, provide a helpful general answer and note the limitation.
- Use numbered steps for procedures. Keep answers concise and structured.
- Do not invent facts not found in the context.\
"""

_ZOOM_SYSTEM_PROMPT = """\
당신은 Zoom 개발자 문서 전문 어시스턴트입니다.

규칙:
1. 제공된 RAG 컨텍스트(Zoom 공식 개발자 문서)만 근거로 답변한다.
2. 근거가 없으면 "현재 수집된 Zoom 공식 문서에서 확인된 근거가 없습니다."라고 답한다.
3. 답변은 한국어로 작성하되, Zoom API/SDK 용어·엔드포인트·파라미터는 영어 원문을 병기한다.
4. 최종 답변에는 공식 출처(source_url, title)를 포함한다.
5. 코드 예시는 마크다운 코드 블록으로 표시한다.\
"""

_CASUAL_SYSTEM_PROMPT = """\
You are a friendly AI assistant for a Korean institution's internal chatbot.
Respond in Korean (한국어) for Korean input.
Be concise, warm, and natural.
You can handle greetings, casual questions, and general conversation.
If the user seems to need Canvas LMS or Panopto help, briefly suggest asking a specific Canvas question.\
"""


def _build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM prompt."""
    if not results:
        return "(No relevant context found.)"
    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r.title or 'Canvas Guide'} — {r.source_url}"
        parts.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(parts)


def _is_analysis_query(query: str) -> bool:
    """기능 요구사항/개발 가능 여부 분석 쿼리인지 감지한다."""
    import re
    if re.search(r'SFR-\d+', query):
        return True
    if query.count('\n- ') >= 3:
        return True
    _FEASIBILITY_KEYWORDS = (
        r'구현\s*가능|개발\s*가능|지원\s*여부|구현\s*여부|개발\s*여부|가능\s*여부'
        r'|기능\s*요구사항|요구사항\s*분석|구현\s*가능한지|개발\s*가능한지'
        r'|지원되는지|지원\s*하는지|가능한지\s*확인|개발\s*가능성|구현\s*가능성'
    )
    return bool(re.search(_FEASIBILITY_KEYWORDS, query))


# keep old name as alias for backward compat with tests
_is_sfr_query = _is_analysis_query


def _build_user_message(query: str, context: str) -> str:
    if _is_analysis_query(query):
        instruction = (
            "아래는 Canvas 생태계 구현 가능성 분석 요청입니다. 반드시 CASE A 형식으로 답변하세요.\n\n"
            "【판정 STRICT 규칙】\n"
            "1. RAG 컨텍스트 [1]~[N]에 명확한 근거가 없으면 절대 ✅로 판정하지 않는다 → 🔍 확인필요\n"
            "2. 모사답안·유사도 검사 → ❌ (Turnitin/Unicheck LTI 필요, Canvas 미내장)\n"
            "3. 동영상 이어보기·10초이동 → ⚠️(b) (Panopto LTI, Canvas LMS 자체 기능 아님)\n"
            "4. 결제·입금계좌·장바구니 → ❌ 또는 ⚠️(c) (Canvas Catalog 또는 외부 결제 필요)\n"
            "5. 근거 번호 [N]은 실제 제공된 컨텍스트 번호만 사용. 임의 [N] 텍스트 사용 금지.\n\n"
            "【출력 형식】\n"
            "- SFR 섹션별로 ## 헤딩 + | 항목 | 판정 | 비고 | 테이블\n"
            "- 비고: [기능 설명] + API: [실제 엔드포인트] + [실제 근거번호] 세 부분 포함\n"
            "- 복합 API: ①POST /api/v1/경로1 → ②GET /api/v1/경로2 [실제번호]\n"
            "- LTI 연동: 'Canvas LMS 자체 기능 아님. [도구명] LTI 연동으로 구현'\n"
            "- 마지막에 ## 요약 테이블(🔍 확인필요 행 포함) + ### 핵심 gap 목록\n\n"
        )
        query = instruction + query
    return f"Context:\n\n{context}\n\nQuestion: {query}"


def _sources_from_results(results: list[SearchResult]) -> list[SourceRef]:
    seen: set[str] = set()
    sources: list[SourceRef] = []
    for r in results:
        if r.source_url not in seen:
            seen.add(r.source_url)
            sources.append(SourceRef(
                title=r.title,
                source_url=r.source_url,
                score=round(r.score, 4),
                source_type="rag",
            ))
    return sources


def _sources_from_web(results: list[WebSearchResult]) -> list[SourceRef]:
    seen: set[str] = set()
    sources: list[SourceRef] = []
    for r in results:
        if r.source_url not in seen:
            seen.add(r.source_url)
            sources.append(SourceRef(
                title=r.title,
                source_url=r.source_url,
                score=round(r.score, 4),
                source_type="web",
            ))
    return sources


class ChatHandler:
    """Orchestrates router → retriever → (web fallback) → LLM for a single chat turn."""

    # Cosine similarity threshold: results below this score are considered
    # off-topic. Kept at 0.50 to accommodate Korean→English cross-lingual
    # queries (bge-m3 scores ~0.53 for correct matches vs ~0.67 for English).
    DEFAULT_MIN_SCORE = 0.58

    def __init__(
        self,
        retriever: CanvasRetriever,
        llm_client: Any,
        llm_model: str,
        router: DomainRouter | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        web_searcher: TavilySearcher | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._model = llm_model
        self._router = router or DomainRouter()
        self._min_score = min_score
        self._web_searcher = web_searcher

    def handle(self, request: ChatRequest) -> ChatResponse:
        decision: RouteDecision = self._router.route(
            request.query, force_domain=request.force_domain
        )

        if not decision.is_canvas:
            return ChatResponse(
                answer=_NOT_CANVAS_ANSWER,
                domain=decision.domain,
                sources=[],
                matched_keywords=decision.matched_keywords,
            )

        # 1. RAG 검색 (CMS/VCMS → product_hint="panopto" 필터 적용)
        product_filter = decision.product_hint
        results = self._retriever.search(
            request.query,
            top_k=request.top_k,
            role=request.role,
            min_score=self._min_score,
            product=product_filter,
        )
        if not results and request.role:
            results = self._retriever.search(
                request.query, top_k=request.top_k, min_score=self._min_score,
                product=product_filter,
            )
        # product 필터로 결과 없으면 전체 검색으로 fallback
        if not results and product_filter:
            results = self._retriever.search(
                request.query, top_k=request.top_k, role=request.role,
                min_score=self._min_score,
            )

        # 2. RAG 신뢰도 확인: 최고 점수 < 0.60이거나 결과 없으면 웹 검색 보조
        _WEB_FALLBACK_THRESHOLD = 0.60
        best_rag_score = max((r.score for r in results), default=0.0)
        web_results: list[WebSearchResult] = []
        if self._web_searcher and best_rag_score < _WEB_FALLBACK_THRESHOLD:
            web_results = self._web_searcher.search(request.query, top_k=request.top_k)

        # 3. 컨텍스트 조합 (RAG + 웹 병합, 또는 웹 단독)
        context_parts = []
        if results:
            context_parts.append(_build_context_block(results))
        if web_results:
            context_parts.append(build_web_context_block(web_results))
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(No relevant context found.)"

        user_msg = _build_user_message(request.query, context)
        completion = self._llm.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        answer = completion.choices[0].message.content or ""

        sources = _sources_from_results(results) + _sources_from_web(web_results)

        return ChatResponse(
            answer=answer,
            domain=decision.domain,
            sources=sources,
            matched_keywords=decision.matched_keywords,
        )


# ---------------------------------------------------------------------------
# Factory — reads env vars
# ---------------------------------------------------------------------------

def build_chat_handler(
    retriever: CanvasRetriever,
    _llm_client: Any = None,
) -> ChatHandler:
    """Build ChatHandler from environment variables."""
    if _llm_client is not None:
        model = os.environ.get("OLLAMA_MODEL", "exaone3.5:7.8b")
        return ChatHandler(retriever=retriever, llm_client=_llm_client, llm_model=model)

    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from openai import OpenAI
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
        model = os.environ.get("OLLAMA_MODEL", "exaone3.5:7.8b")
        client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        from openai import OpenAI
        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    web_searcher = get_web_searcher()

    return ChatHandler(
        retriever=retriever,
        llm_client=client,
        llm_model=model,
        web_searcher=web_searcher,
    )
