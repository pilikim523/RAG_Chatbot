from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


ROLE_PATH_MAP: dict[str, str] = {
    "for-students": "student",
    "for-instructors": "instructor",
    "for-admins": "admin",
    "for-observers": "observer",
    "for-designers": "designer",
    "for-parents": "parent",
}

# All guide collections on community.instructure.com/en/all-guides
# value: (role, sub_product)
CANVAS_COLLECTIONS: dict[str, tuple[str | None, str]] = {
    # Canvas LMS
    "canvas-lms-admin-guide": ("admin", "lms"),
    "canvas-lms-basics-guide": (None, "lms"),
    "canvas-lms-instructor-guide": ("instructor", "lms"),
    "canvas-lms-observer-guide": ("observer", "lms"),
    "canvas-lms-student-guide": ("student", "lms"),
    "canvas-lms-troubleshooting-guide": (None, "lms"),
    "canvas-release-resources": (None, "lms"),
    "canvas-resource": (None, "lms"),
    "canvas-video-guides": (None, "lms"),
    "translated-canvaslms-release-notes": (None, "lms"),
    "integration": (None, "lms"),
    # Canvas Studio
    "canvas-studio-guide": (None, "studio"),
    # Canvas Mobile
    "canvas_mobile-app-android-guide": (None, "mobile"),
    "canvas_mobile-app-ios-guide": (None, "mobile"),
    "canvas_mobile-parent-android-guide": ("parent", "mobile"),
    "canvas_mobile-parent-ios-guide": ("parent", "mobile"),
    "canvas_mobile-teacher-android-guide": ("instructor", "mobile"),
    "canvas_mobile-teacher-ios-guide": ("instructor", "mobile"),
    # Canvas Career
    "canvas_career-guide": (None, "career"),
    # Canvas Catalog
    "canvas_catalog-guide": (None, "catalog"),
    # Canvas Commons
    "canvas_commons-guide": (None, "commons"),
    # Canvas ePortfolios / Pathways
    "canvas-student-pathways-eportfolios-blackboard-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-d2l-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-eportoflios-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-moodlecloud-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-pathways-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-populi-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-sakai-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-schoology-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-talent-guide": ("student", "eportfolio"),
    "canvas-student-pathways-eportfolios-tennessee-guide": ("student", "eportfolio"),
    # Mastery Connect
    "mastery-connect-admin-guide": ("admin", "mastery"),
    "mastery-connect-assessments-guide": (None, "mastery"),
    "mastery-connect-curriculum-maps-guide": (None, "mastery"),
    "mastery-connect-getting-started-guide": (None, "mastery"),
    "mastery-connect-integration-docs": (None, "mastery"),
    "mastery-connect-item-authoring-guide": (None, "mastery"),
    "mastery-connect-reports-guide": (None, "mastery"),
    "mastery-connect-resource-docs": (None, "mastery"),
    "mastery-connect-trackers-guide": (None, "mastery"),
    "mastery-item-bank-guide": (None, "mastery"),
    # Elevate
    "elevate-data-hub-guide": (None, "elevate"),
    "elevate-standards-alignment-alignment-guide": (None, "elevate"),
    "elevate-standards-alignment-app-guide": (None, "elevate"),
    "elevate-standards-alignment-basics-guide": (None, "elevate"),
    "elevate-standards-alignment-reports-guide": (None, "elevate"),
    "elevate-standards-alignment-standards-guide": (None, "elevate"),
    "elevate-standards-alignment-standards-ref": (None, "elevate"),
    "elevate_data_quality-guide": (None, "elevate"),
    "elevate_sa_standards_updates": (None, "elevate"),
    # Parchment
    "parchment-digital-badges-guide": (None, "parchment"),
    "parchment-resource": (None, "parchment"),
    # Portfolium
    "portfolium-resources": (None, "portfolium"),
    # LearnPlatform
    "learnplatform-guide": (None, "learnplatform"),
    # Impact
    "impact-guide": (None, "impact"),
    # IgniteAI
    "igniteai-agent-guide": (None, "igniteai"),
    "igniteai-nutrition-facts": (None, "igniteai"),
    "igniteai-resource": (None, "igniteai"),
    # Intelligent Insights
    "intelligent_insights-guide": (None, "intelligent_insights"),
    # SIS integrations
    "error_codes": (None, "sis"),
    "sis-aeries-guide": (None, "sis"),
    "sis-aspen-guide": (None, "sis"),
    "sis-aspire-guide": (None, "sis"),
    "sis-blackbaud-guide": (None, "sis"),
    "sis-classlink-guide": (None, "sis"),
    "sis-clever-guide": (None, "sis"),
    "sis-data-sync-guide": (None, "sis"),
    "sis-eschoolplus-guide": (None, "sis"),
    "sis-focus-guide": (None, "sis"),
    "sis-genesis-guide": (None, "sis"),
    "sis-infinite-campus-guide": (None, "sis"),
    "sis-peoplesoft-guide": (None, "sis"),
    "sis-pinnacle-guide": (None, "sis"),
    "sis-powerschool-guide": (None, "sis"),
    "sis-progressbook-guide": (None, "sis"),
    "sis-q-guide": (None, "sis"),
    "sis-qmlativ-guide": (None, "sis"),
    "sis-rediker-guide": (None, "sis"),
    "sis-sapphire-guide": (None, "sis"),
    "sis-schooltool-guide": (None, "sis"),
    "sis-skyward-guide": (None, "sis"),
    "sis-sunet-guide": (None, "sis"),
    "sis-synergy-guide": (None, "sis"),
    "sis-veracross-guide": (None, "sis"),
    "sis_additional_resources": (None, "sis"),
    "sis_integration": (None, "sis"),
    # Community
    "instructure-community-guide": (None, "community"),
}


def collection_metadata(slug: str) -> tuple[str | None, str | None]:
    """Return (role, sub_product) for a canvas collection slug, or (None, None)."""
    result = CANVAS_COLLECTIONS.get(slug.lower())
    if result is None:
        return None, None
    return result


class DiscoveryEntry(BaseModel):
    source_url: str
    canonical_url: str | None = None
    title: str | None = None
    product: str = "canvas"
    guide: str | None = None
    category: str | None = None
    role: str | None = None
    discovered_at: datetime
    status: Literal["pending", "fetched", "failed", "skipped"] = "pending"
    content_hash: str | None = None
    fetch_error: str | None = None
    raw_html_path: str | None = None
    retry_count: int = 0

    def to_jsonl_line(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> "DiscoveryEntry":
        return cls.model_validate_json(line)


def load_manifest(path: Path) -> list[DiscoveryEntry]:
    if not path.exists():
        return []
    entries: list[DiscoveryEntry] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(DiscoveryEntry.from_jsonl_line(line))
    return entries


def save_manifest(entries: list[DiscoveryEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_jsonl_line() + "\n")


def sha256_of(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def role_from_path(path: str) -> str | None:
    parts = path.lower().strip("/").split("/")
    for part in parts:
        if part in ROLE_PATH_MAP:
            return ROLE_PATH_MAP[part]
    return None


def guide_from_path(path: str) -> str | None:
    """Return last path segment for guide/article URLs with 4+ segments."""
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) >= 4:
        return parts[-1]
    return None


def category_from_path(path: str) -> str | None:
    """Return category segment for old /en/canvas/for-*/slug style paths."""
    parts = [p for p in path.strip("/").split("/") if p]
    # e.g. ['en', 'canvas', 'for-students', 'slug'] → 'for-students'
    if len(parts) >= 3:
        candidate = parts[2]
        if candidate.startswith("for-") or candidate in ROLE_PATH_MAP:
            return candidate
    return None
