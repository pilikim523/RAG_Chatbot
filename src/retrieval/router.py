"""
Stage 6: Domain router — classifies a user query into a routing domain.

Domains
-------
canvas   : Route to Canvas RAG retriever first.
internal : Route to internal policy/operations documents (future).
general  : No RAG context; answer with LLM general knowledge.

Priority
--------
1. Explicit UI selection (force_domain) — always wins.
2. "canvas" literal present anywhere → canvas.
3. Canvas keyword match (EN or KO) → canvas.
4. Everything else → general  (internal reserved for future integration).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Domain = Literal["canvas", "internal", "general"]

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
})

_CANVAS_KEYWORDS_KO: frozenset[str] = frozenset({
    "캔버스",
    "과제", "성적", "퀴즈", "모듈", "루브릭",
    "강의", "강좌", "수업",
    "학생", "교수자", "교수", "관리자", "강사",
    "토론", "공지", "공지사항", "달력",
    "제출", "출석", "평가",
    "등록", "수강",
    # 개발자/API 관련
    "인증", "액세스 토큰", "엔드포인트", "웹훅",
    "데이터 접근", "데이터 동기화", "페이지네이션",
})

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
    ) -> RouteDecision:
        """Return routing decision for query.

        force_domain: explicit UI selection — overrides keyword detection.
        """
        if force_domain:
            return RouteDecision(domain=force_domain, forced=True)

        matched = _detect_canvas_keywords(query)
        if matched:
            return RouteDecision(domain="canvas", matched_keywords=matched)

        return RouteDecision(domain="general")


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


def route_query(query: str, force_domain: Domain | None = None) -> RouteDecision:
    """Module-level convenience wrapper around DomainRouter."""
    return DomainRouter().route(query, force_domain=force_domain)
