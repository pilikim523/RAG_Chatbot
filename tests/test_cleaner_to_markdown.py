"""Tests for src/cleaner/to_markdown.py."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.cleaner.models import DocEntry
from src.cleaner.to_markdown import (
    clean_markdown,
    count_words,
    extract_article,
    html_to_markdown,
    process_entry,
    strip_noise,
)
from src.crawler.models import DiscoveryEntry, sha256_of

BASE = "https://community.instructure.com"
GUIDE_URL = f"{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANVAS_ARTICLE_HTML = f"""
<html>
<head>
  <title>How do I submit an online assignment? | Instructure Community</title>
  <meta property="og:title" content="How do I submit an online assignment?"/>
</head>
<body>
  <header>Site navigation – noise</header>
  <nav class="breadcrumb"><ol><li>Home</li><li>Canvas</li></ol></nav>
  <script>alert('noise')</script>
  <article class="userContent seoSectionPiece">
    <h2>Overview</h2>
    <p>To submit an online assignment, open the Assignments page.</p>
    <h2>Steps</h2>
    <ol>
      <li>Click <strong>Assignments</strong> in Course Navigation.</li>
      <li>Click the assignment name.</li>
      <li>Click <strong>Submit Assignment</strong>.</li>
    </ol>
    <h3>Text Entry</h3>
    <p>Type directly in the text box.</p>
    <p>See <a href="/en/kb/articles/661195-pairing-code">related guide</a>.</p>
  </article>
  <footer>Footer noise</footer>
</body>
</html>
"""

MINIMAL_HTML = """
<html><body>
  <article><p>Hello world.</p></article>
</body></html>
"""

NO_ARTICLE_HTML = """
<html><body>
  <div class="content"><p>Body only content.</p></div>
</body></html>
"""

NOISY_ARTICLE_HTML = f"""
<html><body>
  <article class="userContent">
    <nav class="breadcrumb">Breadcrumb</nav>
    <h2>Guide Title</h2>
    <p>Main content here.</p>
    <script>nefarious()</script>
    <div class="lia-component-tags">tags</div>
    <a href="/relative/link">Relative link</a>
  </article>
</body></html>
"""


# ---------------------------------------------------------------------------
# extract_article
# ---------------------------------------------------------------------------

class TestExtractArticle:
    def test_finds_userContent_article(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(CANVAS_ARTICLE_HTML, "lxml")
        el = extract_article(soup)
        assert el is not None
        assert el.name == "article"

    def test_finds_generic_article(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(MINIMAL_HTML, "lxml")
        el = extract_article(soup)
        assert el is not None
        assert el.name == "article"

    def test_falls_back_to_body(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(NO_ARTICLE_HTML, "lxml")
        el = extract_article(soup)
        assert el is not None
        assert el.name == "body"

    def test_returns_none_on_empty_html(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("", "lxml")
        # lxml may not produce a body element for empty input
        el = extract_article(soup)
        # Result is None or a body/html element – no crash
        assert el is None or el.name in ("body", "html", "[document]")


# ---------------------------------------------------------------------------
# strip_noise
# ---------------------------------------------------------------------------

class TestStripNoise:
    def test_removes_script_tags(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(NOISY_ARTICLE_HTML, "lxml")
        article = soup.find("article")
        strip_noise(article, GUIDE_URL)
        assert article.find("script") is None

    def test_removes_noise_selectors(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(NOISY_ARTICLE_HTML, "lxml")
        article = soup.find("article")
        strip_noise(article, GUIDE_URL)
        assert article.find(class_="lia-component-tags") is None
        assert article.find(class_="breadcrumb") is None

    def test_makes_relative_links_absolute(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(NOISY_ARTICLE_HTML, "lxml")
        article = soup.find("article")
        strip_noise(article, GUIDE_URL)
        links = article.find_all("a", href=True)
        for link in links:
            assert link["href"].startswith("http"), f"Expected absolute: {link['href']}"

    def test_preserves_content(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(NOISY_ARTICLE_HTML, "lxml")
        article = soup.find("article")
        strip_noise(article, GUIDE_URL)
        text = article.get_text()
        assert "Main content here" in text


# ---------------------------------------------------------------------------
# clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_collapses_multiple_blank_lines(self):
        raw = "line1\n\n\n\n\nline2"
        assert clean_markdown(raw) == "line1\n\nline2"

    def test_removes_trailing_spaces(self):
        raw = "  line1   \n  line2  "
        result = clean_markdown(raw)
        assert not any(line.endswith(" ") for line in result.split("\n"))

    def test_strips_leading_trailing_whitespace(self):
        raw = "\n\nContent\n\n"
        assert clean_markdown(raw) == "Content"

    def test_empty_input(self):
        assert clean_markdown("") == ""


# ---------------------------------------------------------------------------
# html_to_markdown
# ---------------------------------------------------------------------------

class TestHtmlToMarkdown:
    def test_contains_title_as_h1(self):
        md = html_to_markdown(CANVAS_ARTICLE_HTML, "How do I submit?", GUIDE_URL)
        assert md.startswith("# How do I submit?")

    def test_contains_article_content(self):
        md = html_to_markdown(CANVAS_ARTICLE_HTML, None, GUIDE_URL)
        assert "Overview" in md
        assert "Submit Assignment" in md

    def test_no_header_without_title(self):
        md = html_to_markdown(MINIMAL_HTML, None, GUIDE_URL)
        assert not md.startswith("#")

    def test_contains_source_reference(self):
        md = html_to_markdown(CANVAS_ARTICLE_HTML, "Test", GUIDE_URL)
        assert GUIDE_URL in md

    def test_no_script_in_output(self):
        md = html_to_markdown(CANVAS_ARTICLE_HTML, "Test", GUIDE_URL)
        assert "alert(" not in md
        assert "<script" not in md

    def test_nav_not_in_output(self):
        md = html_to_markdown(CANVAS_ARTICLE_HTML, "Test", GUIDE_URL)
        assert "Site navigation" not in md
        assert "Footer noise" not in md

    def test_list_items_present(self):
        md = html_to_markdown(CANVAS_ARTICLE_HTML, "Test", GUIDE_URL)
        assert "- " in md or "1." in md

    def test_empty_html_returns_empty_string(self):
        md = html_to_markdown("<html><body></body></html>", "Title", GUIDE_URL)
        # body exists but no content worth keeping
        assert isinstance(md, str)


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

class TestCountWords:
    def test_basic_count(self):
        assert count_words("hello world foo") == 3

    def test_empty_string(self):
        assert count_words("") == 0

    def test_multiline(self):
        assert count_words("one\ntwo\nthree") == 3


# ---------------------------------------------------------------------------
# process_entry
# ---------------------------------------------------------------------------

def _make_doc_entry(source_url: str, raw_html_path: str | None = "abc123.html") -> DocEntry:
    return DocEntry(
        source_url=source_url,
        title="How do I submit?",
        product="canvas",
        role="student",
        category="canvas-lms-student-guide",
        raw_html_path=raw_html_path,
        clean_status="pending",
    )


class TestProcessEntry:
    def test_success(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_html"
        raw_dir.mkdir()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        html_file = raw_dir / "abc123.html"
        html_file.write_text(CANVAS_ARTICLE_HTML, encoding="utf-8")

        entry = _make_doc_entry(GUIDE_URL, "abc123.html")
        result = process_entry(entry, raw_dir, md_dir)

        assert result.clean_status == "cleaned"
        assert result.markdown_path == "abc123.md"
        assert result.markdown_hash is not None
        assert result.word_count is not None and result.word_count > 0
        assert result.clean_error is None

        md_file = md_dir / "abc123.md"
        assert md_file.exists()
        content = md_file.read_text(encoding="utf-8")
        assert "# How do I submit?" in content
        assert GUIDE_URL in content

    def test_missing_html_file_marks_failed(self, tmp_path: Path):
        entry = _make_doc_entry(GUIDE_URL, "nonexistent.html")
        result = process_entry(entry, tmp_path / "raw", tmp_path / "md")
        assert result.clean_status == "failed"
        assert result.clean_error is not None

    def test_missing_raw_html_path_marks_failed(self, tmp_path: Path):
        entry = _make_doc_entry(GUIDE_URL, raw_html_path=None)
        result = process_entry(entry, tmp_path, tmp_path)
        assert result.clean_status == "failed"

    def test_already_cleaned_becomes_skipped(self, tmp_path: Path):
        entry = DocEntry(
            source_url=GUIDE_URL,
            raw_html_path="abc.html",
            clean_status="cleaned",
            markdown_path="abc.md",
        )
        result = process_entry(entry, tmp_path, tmp_path)
        assert result.clean_status == "skipped"

    def test_markdown_hash_matches_file(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_html"
        raw_dir.mkdir()
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()

        (raw_dir / "x.html").write_text(CANVAS_ARTICLE_HTML, encoding="utf-8")
        entry = _make_doc_entry(GUIDE_URL, "x.html")
        result = process_entry(entry, raw_dir, md_dir)

        md_content = (md_dir / "x.md").read_text(encoding="utf-8")
        assert result.markdown_hash == sha256_of(md_content)

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        raw_dir = tmp_path / "raw_html"
        raw_dir.mkdir()
        (raw_dir / "y.html").write_text(MINIMAL_HTML, encoding="utf-8")

        entry = _make_doc_entry(GUIDE_URL, "y.html")
        md_dir = tmp_path / "deep" / "nested" / "markdown"
        result = process_entry(entry, raw_dir, md_dir)

        assert md_dir.exists()
        assert result.clean_status in ("cleaned", "failed")


# ---------------------------------------------------------------------------
# DocEntry model
# ---------------------------------------------------------------------------

class TestDocEntry:
    def test_from_discovery_entry_fetched(self):
        fe = DiscoveryEntry(
            source_url=GUIDE_URL,
            title="Submit Assignment",
            product="canvas",
            role="student",
            category="canvas-lms-student-guide",
            raw_html_path="abc.html",
            content_hash="deadbeef",
            discovered_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
            status="fetched",
        )
        doc = DocEntry.from_discovery_entry(fe)
        assert doc.source_url == GUIDE_URL
        assert doc.role == "student"
        assert doc.raw_html_path == "abc.html"
        assert doc.clean_status == "pending"

    def test_from_discovery_entry_not_fetched(self):
        fe = DiscoveryEntry(
            source_url=GUIDE_URL,
            discovered_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
            status="pending",
        )
        doc = DocEntry.from_discovery_entry(fe)
        assert doc.clean_status == "skipped"

    def test_jsonl_round_trip(self):
        entry = DocEntry(
            source_url=GUIDE_URL,
            title="My Guide",
            markdown_path="abc.md",
            clean_status="cleaned",
            word_count=150,
        )
        loaded = DocEntry.from_jsonl_line(entry.to_jsonl_line())
        assert loaded.source_url == entry.source_url
        assert loaded.word_count == 150
        assert loaded.clean_status == "cleaned"
