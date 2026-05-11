"""
Stage 6: Domain router — classifies a user query into a routing domain.

Domains
-------
canvas   : Route to Canvas RAG retriever first.
internal : Route to internal policy/operations documents (future).
web      : General web search via SearXNG (non-Canvas factual questions).
casual   : Direct LLM — greetings, meta questions, light conversation.
general  : Legacy alias for web (backward compatibility).

Priority
--------
1. Explicit UI selection (force_domain) — always wins.
2. "canvas" literal present anywhere → canvas.
3. Canvas keyword match (EN or KO) → canvas.
4. CMS / VCMS keyword → canvas + product_hint="panopto" (Canvas Studio override if explicit).
5. Casual pattern match → casual.
6. Everything else → web.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Domain = Literal["canvas", "zoom", "internal", "general", "web", "casual"]

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_CANVAS_KEYWORDS_EN: frozenset[str] = frozenset({
    # Core features
    "canvas", "course", "assignment", "gradebook", "quiz", "module",
    "rubric", "speedgrader", "discussion", "announcement", "page",
    "syllabus", "calendar", "section", "submission", "grade",
    # People / roles
    "enrollment", "instructor", "student", "observer", "admin",
    "teacher", "designer", "ta",
    # Integrations
    "lti", "sis", "commons", "catalog",
    # Sub-products
    "canvas studio", "canvas catalog", "canvas commons",
    # Assignment date terminology
    "due date", "availability date", "available from", "available until",
    "lock date", "unlock date", "late submission", "late policy",
    "peer review", "attempt", "resubmit", "resubmission",
    # Actions common in Canvas context
    "submit", "publish", "unpublish", "mastery", "outcome",
    "blueprint", "import", "export",
    # Developer / API keywords (developerdocs.instructure.com)
    "api", "rest api", "graphql", "oauth", "oauth2", "access token",
    "dap", "data access platform", "datasync", "data sync",
    "ab connect",
    "endpoint", "webhook", "pagination", "rate limit",
    "canvas api", "canvas lms api", "instructure api",
    "parchment", "elevate", "impact", "learnplatform",
    "mastery connect", "mastery item bank",
    "canvas career", "canvas catalog", "canvas commons",
    "swagger", "openapi", "scorm", "xapi",
    # Video CMS (routes to Panopto by default)
    "cms", "vcms", "panopto",
})

_CANVAS_KEYWORDS_KO: frozenset[str] = frozenset({
    "캔버스",
    "과제", "성적", "퀴즈", "모듈", "루브릭",
    "강의", "강좌", "수업",
    "학생", "교수자", "교수", "관리자", "강사",
    "토론", "공지", "공지사항", "달력",
    "제출", "출석", "평가",
    "등록", "수강",
    # 날짜/마감 관련
    "마감일", "마감 날짜", "마감날짜", "제출 마감", "제출마감",
    "이용 가능", "잠금 날짜", "열람 가능", "공개일",
    "지각 제출", "늦은 제출", "재제출",
    "동료 평가", "피어 리뷰",
    # 개발자/API 관련
    "인증", "액세스 토큰", "엔드포인트", "웹훅",
    "데이터 접근", "데이터 동기화", "페이지네이션",
    # Video CMS / Panopto
    "영상관리", "비디오관리",
    "파놉토",
    "자막", "자동자막", "자동 자막", "CC자막",
    "녹화", "녹화물", "녹음", "녹화본",
    "영상", "비디오", "동영상",
    "스트리밍", "라이브",
    "캡처", "화면캡처",
})

_ZOOM_KEYWORDS_EN: frozenset[str] = frozenset({
    "zoom", "zoom api", "zoom sdk", "zoom oauth",
    "zoom meeting", "zoom webinar", "zoom phone", "zoom chat", "zoom rooms",
    "zoom video sdk", "zoom meeting sdk", "zoom app",
    "zoom marketplace", "zoom webhook", "zoom events",
    "zoom recording", "zoom participant", "zoom host",
    "server-to-server oauth", "zoom jwt",
    "zoom rest api", "zoom graphql",
    "meeting id", "zoom link",
})
_ZOOM_KEYWORDS_KO: frozenset[str] = frozenset({
    "줌", "줌 api", "줌 sdk", "줌 oauth",
    "줌 미팅", "줌 웨비나", "줌 폰", "줌 채팅",
    "줌 녹화", "줌 참가자", "줌 호스트",
    "줌 마켓플레이스", "줌 앱",
    "화상회의", "소회의실", "대기실",
})

_ZOOM_EN_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9])("
    + "|".join(re.escape(k) for k in sorted(_ZOOM_KEYWORDS_EN, key=len, reverse=True))
    + r")(?![a-zA-Z0-9])",
    re.IGNORECASE,
)
_ZOOM_KO_PATTERN = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(_ZOOM_KEYWORDS_KO, key=len, reverse=True)) + ")",
)

# CMS/VCMS 키워드 패턴 (단어 경계, 대소문자 무시)
_CMS_PATTERN = re.compile(r"(?<![a-zA-Z0-9])(vcms|cms)(?![a-zA-Z0-9])", re.IGNORECASE)
# Canvas Studio 명시 패턴
_CANVAS_STUDIO_PATTERN = re.compile(r"canvas\s+studio", re.IGNORECASE)

# 일상 대화 / 인사 / 메타 질문 패턴
_CASUAL_PATTERN = re.compile(
    r"^(안녕|hi\b|hello\b|hey\b|반가워|고마워|고맙|감사합니다|감사해|수고|잘가|ㅋㅋ+|ㅎㅎ+|ㅠ+|ㅜ+|ㅇㅇ|넵|오케|ok\b)"
    r"|뭘\s*(할\s*줄|할\s*수|도와\s*줄)"
    r"|너는?\s*(뭐야?|누구야?|어떤|어때)"
    r"|당신은?\s*(누구|뭐야?)"
    r"|무슨\s*일\s*(해|하는)",
    re.IGNORECASE,
)

# 캔버스 대화 이어가기용 후속 질문 패턴 (짧은 참조 표현)
_CANVAS_FOLLOWUP_PATTERN = re.compile(
    r"^(그|이|저|그것|이것|저것|그러면|그래서|그럼|그건|이건|그렇다면|방금|아까|거기서|그\s*방법|위에서|앞에서)"
)

# EN: ASCII word boundary (not preceded/followed by [a-zA-Z0-9]).
# Using \b would treat Korean chars as \w in Python 3 Unicode mode,
# preventing matches like "Canvas에서". ASCII lookaround avoids this.
_EN_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9])("
    + "|".join(re.escape(k) for k in sorted(_CANVAS_KEYWORDS_EN, key=len, reverse=True))
    + r")(?![a-zA-Z0-9])",
    re.IGNORECASE,
)
_KO_PATTERN = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(_CANVAS_KEYWORDS_KO, key=len, reverse=True)) + ")",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    domain: Domain
    matched_keywords: list[str] = field(default_factory=list)
    forced: bool = False        # True when force_domain overrode detection
    product_hint: str | None = None  # "panopto" | "canvas" | None (no filter)

    @property
    def is_canvas(self) -> bool:
        return self.domain == "canvas"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class DomainRouter:
    """Classify a user query to determine which RAG pipeline to invoke."""

    def route(
        self,
        query: str,
        force_domain: Domain | None = None,
        has_canvas_history: bool = False,
    ) -> RouteDecision:
        """Return routing decision for query.

        force_domain: explicit UI selection — overrides keyword detection.
        has_canvas_history: True when the session already has Canvas turns.
          Short follow-up queries in a Canvas session stay routed to Canvas.
        CMS/VCMS → product_hint="panopto" unless "Canvas Studio" is explicitly mentioned.
        """
        if force_domain:
            return RouteDecision(domain=force_domain, forced=True)

        canvas_matched = _detect_canvas_keywords(query)
        zoom_matched = _detect_zoom_keywords(query)
        product_hint = _detect_product_hint(query)

        # Zoom 키워드가 있고 "canvas"/"캔버스"가 명시되지 않으면 zoom 우선.
        # e.g. "zoom api rate limits" → zoom / "Canvas zoom integration" → canvas
        if zoom_matched and canvas_matched:
            has_explicit_canvas = any(k in {"canvas", "캔버스"} for k in canvas_matched)
            if not has_explicit_canvas:
                return RouteDecision(domain="zoom", matched_keywords=zoom_matched)

        if canvas_matched:
            return RouteDecision(
                domain="canvas",
                matched_keywords=canvas_matched,
                product_hint=product_hint,
            )

        if zoom_matched:
            return RouteDecision(domain="zoom", matched_keywords=zoom_matched)

        # Canvas 대화 중 짧은 후속 질문은 Canvas로 유지
        if has_canvas_history and _is_canvas_followup(query):
            return RouteDecision(domain="canvas", matched_keywords=["(이전 대화 컨텍스트)"])

        if _is_casual(query):
            return RouteDecision(domain="casual")

        return RouteDecision(domain="web")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_canvas_keywords(query: str) -> list[str]:
    """Return list of matched Canvas keywords (empty = not a Canvas query)."""
    found: list[str] = []
    for m in _EN_PATTERN.finditer(query):
        found.append(m.group(0).lower())
    for m in _KO_PATTERN.finditer(query):
        found.append(m.group(0))
    # Deduplicate while preserving first-occurrence order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in found:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _detect_zoom_keywords(query: str) -> list[str]:
    """Return list of matched Zoom keywords (empty = not a Zoom query)."""
    found: list[str] = []
    for m in _ZOOM_EN_PATTERN.finditer(query):
        found.append(m.group(0).lower())
    for m in _ZOOM_KO_PATTERN.finditer(query):
        found.append(m.group(0))
    seen: set[str] = set()
    unique: list[str] = []
    for kw in found:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def _detect_product_hint(query: str) -> str | None:
    """CMS/VCMS → 'panopto' (default), unless 'Canvas Studio' is explicitly mentioned."""
    if _CMS_PATTERN.search(query):
        if _CANVAS_STUDIO_PATTERN.search(query):
            return "canvas"
        return "panopto"
    return None


def _is_casual(query: str) -> bool:
    """인사, 메타 질문, 짧은 반응 등 일상 대화 감지."""
    return bool(_CASUAL_PATTERN.search(query.strip()))


def _is_canvas_followup(query: str) -> bool:
    """Canvas 대화 이력이 있을 때 후속 참조 표현 감지 (짧고 지시어로 시작)."""
    q = query.strip()
    return len(q) < 30 and bool(_CANVAS_FOLLOWUP_PATTERN.match(q))


def route_query(query: str, force_domain: Domain | None = None) -> RouteDecision:
    """Module-level convenience wrapper around DomainRouter."""
    return DomainRouter().route(query, force_domain=force_domain)
