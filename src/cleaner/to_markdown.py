"""
Stage 2: Raw HTML → clean Markdown.

Reads fetch manifest (canvas_urls.jsonl or developer_docs_urls.jsonl, entries
with status=fetched), loads raw HTML from raw_html_dir, extracts article
content, converts to Markdown, writes .md files to out_dir, and writes a
docs manifest JSONL.

Content extraction strategy:

community.instructure.com (Canvas Guides):
  1. <article class="userContent …">  (Instructure Community KB articles)
  2. <article>                         (generic fallback)
  3. <body>                            (last resort)

developerdocs.instructure.com (GitBook):
  1. [data-testid="page.contentEditor"]  (GitBook primary content pane)
  2. <main>                              (GitBook semantic main)
  3. <article>                           (generic fallback)
  4. <body>                              (last resort)

developers.zoom.us (Next.js SSG with compiled MDX):
  1. __NEXT_DATA__ JSON → pageProps.mainContent.code (compiled JSX)
     Text extracted from JSX children string literals with heading detection.
"""
from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import click
from bs4 import BeautifulSoup
from markdownify import markdownify
from rich.console import Console

from src.crawler.models import DiscoveryEntry, load_manifest, sha256_of
from src.cleaner.models import DocEntry, load_doc_manifest, save_doc_manifest

console = Console(stderr=True)

# Tags to strip before converting to Markdown (noise inside article element)
_NOISE_TAGS = [
    "script", "style", "nav", "noscript", "iframe",
    "button", "form", "svg",
]

# CSS selectors to remove from article content
_NOISE_SELECTORS = [
    # Instructure Community / Lithium noise
    ".breadcrumb", ".lia-navigation-breadcrumbs",
    ".lia-component-tags", ".lia-message-related-content",
    ".lia-feedback-toggle", ".lia-button-group",
    "[aria-hidden='true']",
    ".social-share", ".page-footer", ".site-footer",
    # Zoom developer docs navigation noise
    ".sidebar", ".left-nav", ".right-nav", ".docs-sidebar",
    ".breadcrumbs", ".feedback-section", ".page-navigation",
    # GitBook navigation / chrome noise
    "nav", "aside", "header", "footer",
    "[data-testid='page.desktopTableOfContents']",
    "[data-testid='space.navigation']",
    "[data-testid='page.header']",
    ".gitbook-sidebar", ".gitbook-header",
]

# Content element selectors, tried in order.
# GitBook selectors come first so they win on developer docs pages;
# community.instructure.com selectors follow as fallbacks.
_CONTENT_SELECTORS = [
    # GitBook primary content pane
    "[data-testid='page.contentEditor']",
    # Zoom developer docs (developers.zoom.us)
    ".docs-page-content",
    ".markdown-body",
    ".docs-content",
    "[data-testid='page-content']",
    ".content-wrapper",
    # Generic semantic elements (work for both GitBook <main> and community <article>)
    "article.userContent",
    "article.seoSectionPiece",
    "main",
    "article",
    "[role='main']",
    ".article-body",
    ".page-content",
    "#content",
]


# ---------------------------------------------------------------------------
# Core extraction helpers (pure / testable)
# ---------------------------------------------------------------------------

def extract_article(soup: BeautifulSoup) -> BeautifulSoup | None:
    """Return the primary content element from a page soup."""
    for selector in _CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el:
            return el
    return soup.find("body")


def strip_noise(el: BeautifulSoup, source_url: str) -> None:
    """Remove noise elements and make links absolute – mutates el in place."""
    # Remove noisy tags
    for tag_name in _NOISE_TAGS:
        for tag in el.find_all(tag_name):
            tag.decompose()

    # Remove noisy CSS selectors
    for sel in _NOISE_SELECTORS:
        for tag in el.select(sel):
            tag.decompose()

    # Make relative links absolute
    for a in el.find_all("a", href=True):
        href = str(a["href"]).strip()
        if href and not href.startswith(("http://", "https://", "mailto:", "#")):
            a["href"] = urljoin(source_url, href)

    # Make relative image src absolute
    for img in el.find_all("img", src=True):
        src = str(img["src"]).strip()
        if src and not src.startswith(("http://", "https://", "data:")):
            img["src"] = urljoin(source_url, src)


def clean_markdown(raw: str) -> str:
    """Normalize whitespace and remove Markdown noise artifacts."""
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", raw)
    # Remove trailing spaces on lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Remove lines that are only dashes/underscores (HR noise from markdownify)
    text = re.sub(r"\n[-_]{3,}\n", "\n\n---\n\n", text)
    return text.strip()


def extract_nextjs_mdx(html_str: str) -> str | None:
    """Extract text content from Next.js pages with compiled MDX in __NEXT_DATA__.

    developers.zoom.us pages embed compiled JSX in pageProps.mainContent.code.
    We extract string literals with heading structure detection.
    Returns Markdown string or None if not a Next.js MDX page.
    """
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html_str, re.DOTALL,
    )
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        mc = data.get("props", {}).get("pageProps", {}).get("mainContent")
        if not mc or not isinstance(mc, dict):
            return None
        code = mc.get("code", "")
        frontmatter = mc.get("frontmatter", {})
        if not code:
            return None
    except (json.JSONDecodeError, AttributeError):
        return None

    lines: list[str] = []

    # Frontmatter description as intro
    if frontmatter.get("description"):
        lines.append(frontmatter["description"])
        lines.append("")

    # Walk through code extracting heading/paragraph content in order.
    # Pattern: jsx element tag followed by children string literal.
    # e.g. (0,n.jsx)(e.h2,{...children:"Heading Text"...})
    #      (0,n.jsx)(e.p,{children:"Paragraph text"})
    token_re = re.compile(
        r'e\.(h[1-6]|p|li|td|th|strong|code)(?:,|\b).*?children:"((?:[^"\\]|\\.){5,500})"',
        re.DOTALL,
    )

    seen: set[str] = set()
    for match in token_re.finditer(code):
        tag = match.group(1)
        text = match.group(2).replace('\\"', '"').replace("\\n", " ").strip()
        if not text or text in seen:
            continue
        # Skip short fragments and strings without spaces (code tokens / IDs)
        is_heading = tag.startswith("h")
        min_len = 8 if is_heading else 25
        if len(text) < min_len:
            continue
        # Skip strings that look like code identifiers (no spaces, camelCase etc.)
        if " " not in text and not is_heading:
            continue
        seen.add(text)
        if tag == "h1":
            lines.append(f"# {text}")
        elif tag == "h2":
            lines.append(f"## {text}")
        elif tag == "h3":
            lines.append(f"### {text}")
        elif tag in ("h4", "h5", "h6"):
            lines.append(f"#### {text}")
        elif tag == "li":
            lines.append(f"- {text}")
        elif tag == "code":
            lines.append(f"`{text}`")
        else:
            lines.append(text)

    if not lines:
        return None
    return "\n".join(lines)


def html_to_markdown(html_str: str, title: str | None, source_url: str) -> str:
    """Full pipeline: parse → extract → strip → convert → clean → wrap."""
    # Try Next.js MDX extraction first (developers.zoom.us)
    nextjs_content = extract_nextjs_mdx(html_str)
    if nextjs_content:
        parts: list[str] = []
        if title:
            parts.append(f"# {title}")
            parts.append("")
        parts.append(nextjs_content)
        parts.append("")
        parts.append("---")
        parts.append(f"**Source:** <{source_url}>")
        return clean_markdown("\n".join(parts))

    soup = BeautifulSoup(html_str, "lxml")
    article = extract_article(soup)
    if article is None:
        return ""
    strip_noise(article, source_url)

    body_md = markdownify(
        str(article),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style"],
    )
    body_md = clean_markdown(body_md)

    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
        parts.append("")
    parts.append(body_md)
    parts.append("")
    parts.append("---")
    parts.append(f"**Source:** <{source_url}>")

    return "\n".join(parts)


def count_words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Process a single manifest entry
# ---------------------------------------------------------------------------

def process_entry(
    entry: DocEntry,
    raw_html_dir: Path,
    out_dir: Path,
) -> DocEntry:
    """Convert one HTML file to Markdown. Returns updated DocEntry."""
    if entry.clean_status == "cleaned":
        return entry.model_copy(update={"clean_status": "skipped"})

    if not entry.raw_html_path:
        return entry.model_copy(update={
            "clean_status": "failed",
            "clean_error": "raw_html_path is missing",
        })

    html_path = raw_html_dir / entry.raw_html_path
    if not html_path.exists():
        return entry.model_copy(update={
            "clean_status": "failed",
            "clean_error": f"HTML file not found: {html_path}",
        })

    try:
        html = html_path.read_text(encoding="utf-8")
        markdown = html_to_markdown(html, entry.title, entry.source_url)
        if not markdown.strip():
            return entry.model_copy(update={
                "clean_status": "failed",
                "clean_error": "empty markdown output",
            })

        md_filename = html_path.stem + ".md"
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / md_filename
        md_path.write_text(markdown, encoding="utf-8")

        return entry.model_copy(update={
            "clean_status": "cleaned",
            "markdown_path": md_filename,
            "markdown_hash": sha256_of(markdown),
            "word_count": count_words(markdown),
            "clean_error": None,
            "cleaned_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        return entry.model_copy(update={
            "clean_status": "failed",
            "clean_error": str(exc),
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--input-manifest", required=True, type=click.Path(exists=True),
    help="Fetch manifest (canvas_urls.jsonl with status=fetched entries)",
)
@click.option(
    "--raw-html-dir", default="data/raw_html", show_default=True, type=click.Path(),
    help="Base directory containing raw HTML files",
)
@click.option(
    "--out-dir", required=True, type=click.Path(),
    help="Output directory for Markdown files",
)
@click.option(
    "--out-manifest", required=True, type=click.Path(),
    help="Output DocEntry manifest path (canvas_docs.jsonl)",
)
@click.option("--dry-run", is_flag=True, help="Print what would be processed without writing")
def main(
    input_manifest: str,
    raw_html_dir: str,
    out_dir: str,
    out_manifest: str,
    dry_run: bool,
) -> None:
    """Convert fetched Canvas guide HTML to clean Markdown."""
    raw_dir = Path(raw_html_dir)
    md_dir = Path(out_dir)
    out_path = Path(out_manifest)

    # Load fetch manifest
    fetch_entries: list[DiscoveryEntry] = load_manifest(Path(input_manifest))
    fetched = [e for e in fetch_entries if e.status == "fetched"]
    console.print(
        f"[blue]{len(fetch_entries)} total entries, {len(fetched)} fetched → cleaning[/blue]"
    )

    # Load existing doc manifest for resume support
    existing = load_doc_manifest(out_path)
    existing_urls = {e.source_url for e in existing if e.clean_status == "cleaned"}

    # Build DocEntry list from fetch manifest
    doc_entries: list[DocEntry] = []
    for fe in fetched:
        if fe.source_url in existing_urls:
            # Already cleaned – carry forward
            existing_entry = next(e for e in existing if e.source_url == fe.source_url)
            doc_entries.append(existing_entry.model_copy(update={"clean_status": "skipped"}))
        else:
            doc_entries.append(DocEntry.from_discovery_entry(fe))

    if dry_run:
        console.print("[yellow]--dry-run: not writing files[/yellow]")
        pending = [e for e in doc_entries if e.clean_status == "pending"]
        for e in pending:
            console.print(f"  {e.source_url}")
        return

    md_dir.mkdir(parents=True, exist_ok=True)

    results: list[DocEntry] = []
    pending = [e for e in doc_entries if e.clean_status == "pending"]
    skipped = [e for e in doc_entries if e.clean_status != "pending"]

    for i, entry in enumerate(pending, 1):
        result = process_entry(entry, raw_dir, md_dir)
        results.append(result)
        color = "green" if result.clean_status == "cleaned" else "red" if result.clean_status == "failed" else "dim"
        label = result.clean_status
        if result.clean_status == "cleaned":
            label += f" ({result.word_count}w)"
        console.print(f"[{color}][{i}/{len(pending)}] {label}: {entry.source_url}[/{color}]")

    all_entries = skipped + results
    save_doc_manifest(all_entries, out_path)

    cleaned = sum(1 for e in all_entries if e.clean_status in ("cleaned", "skipped") and e.clean_status != "skipped")
    failed = sum(1 for e in all_entries if e.clean_status == "failed")
    console.print(f"[bold]Done: {cleaned} cleaned, {failed} failed → {out_path}[/bold]")


if __name__ == "__main__":
    main()
