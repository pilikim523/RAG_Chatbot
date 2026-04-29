from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.crawler.models import DiscoveryEntry, sha256_of


class DocEntry(BaseModel):
    """One cleaned document: a Canvas guide article with markdown metadata."""

    # Identity / metadata (carried from DiscoveryEntry)
    source_url: str
    canonical_url: str | None = None
    title: str | None = None
    product: str = "canvas"
    guide: str | None = None
    category: str | None = None
    role: str | None = None

    # From fetch stage
    raw_html_path: str | None = None
    content_hash: str | None = None  # SHA-256 of raw HTML

    # From cleaner stage
    markdown_path: str | None = None
    markdown_hash: str | None = None  # SHA-256 of markdown text
    word_count: int | None = None
    clean_status: Literal["pending", "cleaned", "failed", "skipped"] = "pending"
    clean_error: str | None = None
    cleaned_at: datetime | None = None

    def to_jsonl_line(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> "DocEntry":
        return cls.model_validate_json(line)

    @classmethod
    def from_discovery_entry(cls, entry: DiscoveryEntry) -> "DocEntry":
        return cls(
            source_url=entry.source_url,
            canonical_url=entry.canonical_url,
            title=entry.title,
            product=entry.product,
            guide=entry.guide,
            category=entry.category,
            role=entry.role,
            raw_html_path=entry.raw_html_path,
            content_hash=entry.content_hash,
            clean_status="pending" if entry.status == "fetched" else "skipped",
        )


def load_doc_manifest(path: Path) -> list[DocEntry]:
    if not path.exists():
        return []
    entries: list[DocEntry] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(DocEntry.from_jsonl_line(line))
    return entries


def save_doc_manifest(entries: list[DocEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_jsonl_line() + "\n")
