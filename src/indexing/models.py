from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from src.crawler.models import sha256_of


class Chunk(BaseModel):
    """One text chunk ready for embedding and Qdrant ingestion."""

    # Deterministic ID: sha256(source_url + ":" + str(chunk_index))[:16]
    chunk_id: str
    source_url: str
    canonical_url: str | None = None
    title: str | None = None
    product: str = "canvas"
    guide: str | None = None
    category: str | None = None
    role: str | None = None

    chunk_index: int       # 0-based within the source document
    chunk_total: int       # total chunks for this document

    text: str
    char_count: int
    word_count: int
    content_hash: str      # sha256 of chunk text

    def to_jsonl_line(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> "Chunk":
        return cls.model_validate_json(line)


def make_chunk_id(source_url: str, chunk_index: int) -> str:
    return sha256_of(f"{source_url}:{chunk_index}")[:16]


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_jsonl_line(line))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl_line() + "\n")
