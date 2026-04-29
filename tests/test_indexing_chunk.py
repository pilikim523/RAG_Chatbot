"""Tests for src/indexing/chunk.py and src/indexing/models.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.indexing.chunk import (
    _word_count,
    chunk_markdown,
    chunks_for_doc,
    split_long_block,
    split_markdown_blocks,
)
from src.indexing.models import Chunk, load_chunks, make_chunk_id, save_chunks
from src.cleaner.models import DocEntry
from src.crawler.models import sha256_of

BASE = "https://community.instructure.com"
GUIDE_URL = f"{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHORT_MD = """\
# How do I submit?

Click **Assignments** in the Course Navigation menu.

Then click the assignment name.

Click **Submit Assignment** to open the submission form.
"""

LONG_MD = "\n\n".join(
    f"Paragraph {i}: " + "word " * 50  # ~50 words per paragraph
    for i in range(30)
)  # 30 * 50 = 1500 words total

CODE_BLOCK_MD = """\
# Guide

Some introductory text here.

```python
def hello():
    print("world")
    return True
```

More content after the code block.
"""

MULTI_SECTION_MD = """\
# Canvas Assignment Submission

## Overview

Canvas allows students to submit assignments online through the Assignments tool.

## Step-by-step

1. Click Assignments in the Course Navigation.
2. Select the assignment you want to submit.
3. Click Submit Assignment.

## File Upload

You can upload files up to 500 MB.
Supported formats include PDF, DOCX, and ZIP.

## Text Entry

Type your response directly in the text editor.
Format using bold, italic, or bullet lists.
"""


# ---------------------------------------------------------------------------
# split_markdown_blocks
# ---------------------------------------------------------------------------

class TestSplitMarkdownBlocks:
    def test_splits_on_blank_lines(self):
        blocks = split_markdown_blocks("Para 1\n\nPara 2\n\nPara 3")
        assert len(blocks) == 3
        assert blocks[0] == "Para 1"
        assert blocks[1] == "Para 2"

    def test_keeps_code_block_atomic(self):
        blocks = split_markdown_blocks(CODE_BLOCK_MD)
        code_blocks = [b for b in blocks if b.startswith("```")]
        assert len(code_blocks) == 1
        assert "def hello" in code_blocks[0]

    def test_code_block_not_split_by_blank_lines(self):
        md = "Before\n\n```\nline1\n\nline2\n```\n\nAfter"
        blocks = split_markdown_blocks(md)
        code_blocks = [b for b in blocks if "line1" in b]
        assert len(code_blocks) == 1
        assert "line2" in code_blocks[0]

    def test_empty_string_returns_empty(self):
        assert split_markdown_blocks("") == []

    def test_single_paragraph(self):
        blocks = split_markdown_blocks("Just one paragraph")
        assert len(blocks) == 1

    def test_headings_are_separate_blocks(self):
        md = "# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph."
        blocks = split_markdown_blocks(md)
        assert any("# Title" in b for b in blocks)
        assert any("## Section" in b for b in blocks)

    def test_filters_empty_blocks(self):
        blocks = split_markdown_blocks("A\n\n\n\nB")
        assert all(b.strip() for b in blocks)


# ---------------------------------------------------------------------------
# split_long_block
# ---------------------------------------------------------------------------

class TestSplitLongBlock:
    def test_splits_by_sentence(self):
        block = "First sentence. Second sentence. Third sentence. Fourth sentence."
        parts = split_long_block(block, max_words=4)
        assert len(parts) > 1

    def test_does_not_split_short_block(self):
        block = "Short text here."
        parts = split_long_block(block, max_words=100)
        assert len(parts) == 1

    def test_each_part_under_max_words(self):
        block = ". ".join(["word " * 10] * 5)
        parts = split_long_block(block, max_words=15)
        for p in parts:
            assert _word_count(p) <= 20  # some tolerance

    def test_preserves_all_content(self):
        block = "Alpha. Beta. Gamma. Delta. Epsilon."
        parts = split_long_block(block, max_words=2)
        rejoined = " ".join(parts)
        for word in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
            assert word in rejoined


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------

class TestChunkMarkdown:
    def test_empty_text_returns_empty(self):
        assert chunk_markdown("") == []

    def test_short_text_one_chunk(self):
        chunks = chunk_markdown(SHORT_MD, chunk_size=900, chunk_overlap=120)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        chunks = chunk_markdown(LONG_MD, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_size(self):
        chunks = chunk_markdown(LONG_MD, chunk_size=200, chunk_overlap=30)
        for chunk in chunks:
            # Allow slight overflow only for single oversized block
            assert _word_count(chunk) <= 250

    def test_content_coverage(self):
        """All words in the original text appear in at least one chunk."""
        md = "Alpha Beta.\n\nGamma Delta.\n\nEpsilon Zeta."
        chunks = chunk_markdown(md, chunk_size=5, chunk_overlap=1)
        all_text = " ".join(chunks)
        for word in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]:
            assert word in all_text

    def test_overlap_present(self):
        """Last word of chunk N appears in chunk N+1."""
        chunks = chunk_markdown(LONG_MD, chunk_size=100, chunk_overlap=50)
        if len(chunks) >= 2:
            # Get last paragraph of chunk 0 – it should appear in chunk 1
            chunk0_last = chunks[0].split("\n\n")[-1].split()[-5:]  # last 5 words
            assert any(word in chunks[1] for word in chunk0_last)

    def test_preserves_markdown_formatting(self):
        chunks = chunk_markdown(MULTI_SECTION_MD, chunk_size=50, chunk_overlap=10)
        all_text = " ".join(chunks)
        assert "## Overview" in all_text or "Overview" in all_text
        assert "1. Click" in all_text or "Click Assignments" in all_text

    def test_code_block_kept_intact(self):
        chunks = chunk_markdown(CODE_BLOCK_MD, chunk_size=50, chunk_overlap=5)
        code_chunks = [c for c in chunks if "```" in c]
        assert len(code_chunks) >= 1
        # The code block should not be split across chunks
        for c in code_chunks:
            assert c.count("```") % 2 == 0  # balanced fence markers

    def test_single_block_larger_than_chunk_size(self):
        # Single paragraph > chunk_size
        big_para = "word " * 100
        chunks = chunk_markdown(big_para, chunk_size=30, chunk_overlap=5)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

class TestChunkModel:
    def test_make_chunk_id_deterministic(self):
        assert make_chunk_id(GUIDE_URL, 0) == make_chunk_id(GUIDE_URL, 0)

    def test_make_chunk_id_different_for_different_index(self):
        assert make_chunk_id(GUIDE_URL, 0) != make_chunk_id(GUIDE_URL, 1)

    def test_make_chunk_id_length(self):
        assert len(make_chunk_id(GUIDE_URL, 0)) == 16

    def test_jsonl_round_trip(self):
        chunk = Chunk(
            chunk_id=make_chunk_id(GUIDE_URL, 0),
            source_url=GUIDE_URL,
            title="Submit Assignment",
            product="canvas",
            role="student",
            category="canvas-lms-student-guide",
            chunk_index=0,
            chunk_total=3,
            text="Some text here.",
            char_count=15,
            word_count=3,
            content_hash=sha256_of("Some text here."),
        )
        loaded = Chunk.from_jsonl_line(chunk.to_jsonl_line())
        assert loaded.chunk_id == chunk.chunk_id
        assert loaded.chunk_total == 3
        assert loaded.role == "student"

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        chunks = [
            Chunk(
                chunk_id=make_chunk_id(GUIDE_URL, i),
                source_url=GUIDE_URL,
                chunk_index=i, chunk_total=3,
                text=f"Chunk {i} text.",
                char_count=12, word_count=3,
                content_hash=sha256_of(f"Chunk {i}"),
            )
            for i in range(3)
        ]
        path = tmp_path / "chunks.jsonl"
        save_chunks(chunks, path)
        loaded = load_chunks(path)
        assert len(loaded) == 3
        assert loaded[1].chunk_index == 1

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        assert load_chunks(tmp_path / "nope.jsonl") == []

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "chunks.jsonl"
        save_chunks([], path)
        assert path.exists()


# ---------------------------------------------------------------------------
# chunks_for_doc
# ---------------------------------------------------------------------------

class TestChunksForDoc:
    def _entry(self, markdown_path: str | None) -> DocEntry:
        return DocEntry(
            source_url=GUIDE_URL,
            title="Submit Assignment",
            product="canvas",
            role="student",
            category="canvas-lms-student-guide",
            guide="661210-submit",
            markdown_path=markdown_path,
            clean_status="cleaned",
        )

    def test_basic(self, tmp_path: Path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "abc.md").write_text(MULTI_SECTION_MD, encoding="utf-8")

        entry = self._entry("abc.md")
        chunks = chunks_for_doc(entry, md_dir, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 0
        assert all(c.source_url == GUIDE_URL for c in chunks)
        assert all(c.role == "student" for c in chunks)
        assert all(c.chunk_total == len(chunks) for c in chunks)

    def test_chunk_ids_unique(self, tmp_path: Path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "x.md").write_text(LONG_MD, encoding="utf-8")

        entry = self._entry("x.md")
        chunks = chunks_for_doc(entry, md_dir, chunk_size=100, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_sequential(self, tmp_path: Path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "y.md").write_text(LONG_MD, encoding="utf-8")

        entry = self._entry("y.md")
        chunks = chunks_for_doc(entry, md_dir, chunk_size=100, chunk_overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_missing_markdown_path_returns_empty(self, tmp_path: Path):
        entry = self._entry(None)
        result = chunks_for_doc(entry, tmp_path, chunk_size=900, chunk_overlap=120)
        assert result == []

    def test_missing_file_returns_empty(self, tmp_path: Path):
        entry = self._entry("nonexistent.md")
        result = chunks_for_doc(entry, tmp_path, chunk_size=900, chunk_overlap=120)
        assert result == []

    def test_content_hash_matches_text(self, tmp_path: Path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "z.md").write_text(SHORT_MD, encoding="utf-8")

        entry = self._entry("z.md")
        chunks = chunks_for_doc(entry, md_dir, chunk_size=900, chunk_overlap=120)
        for c in chunks:
            assert c.content_hash == sha256_of(c.text)

    def test_word_count_matches_text(self, tmp_path: Path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "wc.md").write_text(SHORT_MD, encoding="utf-8")

        entry = self._entry("wc.md")
        chunks = chunks_for_doc(entry, md_dir, chunk_size=900, chunk_overlap=120)
        for c in chunks:
            assert c.word_count == _word_count(c.text)
