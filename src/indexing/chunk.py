"""
Stage 3: Markdown → overlapping text chunks.

Reads canvas_docs.jsonl (cleaned entries), loads each .md file,
splits into word-count-bounded overlapping chunks, and writes
canvas_chunks.jsonl (one Chunk per line).

Chunking strategy
-----------------
1. Split Markdown into atomic blocks:
   - Fenced code blocks (``` / ~~~) are kept whole.
   - Other paragraphs are separated by blank lines.
2. Expand any single block that exceeds chunk_size via sentence splitting.
3. Greedily accumulate blocks into a chunk until chunk_size words is reached.
4. On flush: carry backwards through the just-emitted blocks to collect
   ~chunk_overlap words of context for the next chunk.
"""
from __future__ import annotations

import re
from pathlib import Path

import click
from rich.console import Console

from src.cleaner.models import DocEntry, load_doc_manifest
from src.crawler.models import sha256_of
from src.indexing.models import Chunk, make_chunk_id, save_chunks

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Text splitting helpers (pure / testable)
# ---------------------------------------------------------------------------

def split_markdown_blocks(text: str) -> list[str]:
    """Split Markdown into paragraph-level blocks.

    Fenced code blocks (``` or ~~~) are kept as single atomic blocks.
    All other content is split on blank lines.
    """
    result: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        stripped = line.strip()

        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            # Flush any pending paragraph first
            if buffer:
                result.append("\n".join(buffer))
                buffer = []
            in_fence = True
            fence_marker = stripped[:3]
            buffer.append(line)

        elif in_fence:
            buffer.append(line)
            if stripped.startswith(fence_marker) and stripped != fence_marker:
                # End of fence (closing marker may have language tag on open)
                pass
            elif stripped == fence_marker:
                # Closing marker – flush code block
                in_fence = False
                fence_marker = ""
                result.append("\n".join(buffer))
                buffer = []

        elif stripped == "":
            if buffer:
                result.append("\n".join(buffer))
                buffer = []

        else:
            buffer.append(line)

    if buffer:
        result.append("\n".join(buffer))

    return [b for b in result if b.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def split_long_block(block: str, max_words: int) -> list[str]:
    """Sentence-split a block that exceeds max_words.

    Falls back to hard word-boundary splitting when no sentence
    boundaries are present (e.g. long lists or repetitive content).
    """
    sentences = re.split(r"(?<=[.!?])\s+", block)

    # No sentence boundaries found – split on word boundaries instead
    if len(sentences) == 1 and _word_count(block) > max_words:
        words = block.split()
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    parts: list[str] = []
    current: list[str] = []
    current_count = 0

    for sent in sentences:
        sc = _word_count(sent)
        if current_count + sc > max_words and current:
            parts.append(" ".join(current))
            current = [sent]
            current_count = sc
        else:
            current.append(sent)
            current_count += sc

    if current:
        parts.append(" ".join(current))

    return parts if parts else [block]


def chunk_markdown(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[str]:
    """Split Markdown into overlapping chunks bounded by word count.

    Returns list of chunk strings (Markdown formatting preserved).
    """
    if not text.strip():
        return []

    # Step 1: atomic blocks
    blocks = split_markdown_blocks(text)

    # Step 2: expand blocks that exceed chunk_size
    expanded: list[str] = []
    for block in blocks:
        if _word_count(block) > chunk_size:
            expanded.extend(split_long_block(block, chunk_size))
        else:
            expanded.append(block)

    if not expanded:
        return []

    # Step 3: greedy accumulation with overlap
    chunks: list[str] = []
    current_blocks: list[str] = []
    current_count = 0

    for block in expanded:
        block_count = _word_count(block)

        if current_count + block_count > chunk_size and current_blocks:
            # Emit chunk
            chunks.append("\n\n".join(current_blocks))

            # Compute overlap: walk backwards until we collect ~chunk_overlap words
            overlap_blocks: list[str] = []
            overlap_count = 0
            for b in reversed(current_blocks):
                bc = _word_count(b)
                if overlap_count + bc <= chunk_overlap:
                    overlap_blocks.insert(0, b)
                    overlap_count += bc
                else:
                    break

            current_blocks = overlap_blocks + [block]
            current_count = overlap_count + block_count
        else:
            current_blocks.append(block)
            current_count += block_count

    if current_blocks:
        chunks.append("\n\n".join(current_blocks))

    return chunks


# ---------------------------------------------------------------------------
# Document → Chunk list
# ---------------------------------------------------------------------------

def chunks_for_doc(entry: DocEntry, markdown_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Load a cleaned Markdown file and return Chunk objects."""
    if not entry.markdown_path:
        return []

    md_path = markdown_dir / entry.markdown_path
    if not md_path.exists():
        console.print(f"[yellow]  WARN: markdown file not found: {md_path}[/yellow]")
        return []

    text = md_path.read_text(encoding="utf-8")
    raw_chunks = chunk_markdown(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total = len(raw_chunks)

    result: list[Chunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        result.append(
            Chunk(
                chunk_id=make_chunk_id(entry.source_url, idx),
                source_url=entry.source_url,
                canonical_url=entry.canonical_url,
                title=entry.title,
                product=entry.product,
                guide=entry.guide,
                category=entry.category,
                role=entry.role,
                chunk_index=idx,
                chunk_total=total,
                text=chunk_text,
                char_count=len(chunk_text),
                word_count=_word_count(chunk_text),
                content_hash=sha256_of(chunk_text),
            )
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--manifest", required=True, type=click.Path(exists=True),
    help="Cleaned doc manifest (canvas_docs.jsonl)",
)
@click.option(
    "--markdown-dir", default="data/markdown", show_default=True, type=click.Path(),
    help="Directory containing Markdown files",
)
@click.option(
    "--out", required=True, type=click.Path(),
    help="Output JSONL path (canvas_chunks.jsonl)",
)
@click.option("--chunk-size", default=900, show_default=True, help="Max words per chunk")
@click.option("--chunk-overlap", default=120, show_default=True, help="Overlap words between chunks")
@click.option("--dry-run", is_flag=True, help="Print stats without writing")
def main(
    manifest: str,
    markdown_dir: str,
    out: str,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
) -> None:
    """Split cleaned Canvas guide Markdown files into overlapping chunks."""
    doc_entries: list[DocEntry] = load_doc_manifest(Path(manifest))
    cleaned = [e for e in doc_entries if e.clean_status == "cleaned"]
    console.print(
        f"[blue]{len(doc_entries)} total entries, {len(cleaned)} cleaned → chunking[/blue]"
    )

    md_dir = Path(markdown_dir)
    all_chunks: list[Chunk] = []

    for i, entry in enumerate(cleaned, 1):
        doc_chunks = chunks_for_doc(entry, md_dir, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)
        console.print(
            f"[dim][{i}/{len(cleaned)}] {len(doc_chunks)} chunk(s): {entry.source_url}[/dim]"
        )

    console.print(
        f"[bold]Total: {len(all_chunks)} chunks from {len(cleaned)} document(s)[/bold]"
    )

    if dry_run:
        console.print("[yellow]--dry-run: not writing output[/yellow]")
        return

    save_chunks(all_chunks, Path(out))
    console.print(f"[green]Saved → {out}[/green]")


if __name__ == "__main__":
    main()
