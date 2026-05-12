"""
URL discovery for Panopto support articles from support.panopto.com.

Panopto's support site is built on Salesforce Experience Cloud, which relies
heavily on JavaScript rendering.  Direct HTTP requests often receive a thin
HTML shell with no rendered article links.

Strategy (in order):
  1. Try https://support.panopto.com/sitemap.xml
     – If found and parseable, collect all /s/article/ and /s/topic/ URLs.
     – If the root sitemap is a sitemap index, fetch each child sitemap.
  2. If sitemap fetch fails or yields no valid URLs, fall back to BFS seeded
     from a curated list of known Panopto support article URLs plus the
     /s/ home page.  BFS depth is capped at 3 to avoid unbounded crawl.

URL acceptance rules (hard):
  - Hostname must be ``support.panopto.com``
  - Path must start with ``/s/article/`` or ``/s/topic/``

For each accepted URL:
  - ``guide``    = last non-empty path segment (slug portion before any ``?``)
  - ``category`` = path segment immediately before the guide (e.g. ``article``
                   or ``topic``)
  - ``role``     = None  (Panopto docs are not role-segmented in the URL)

Fetch failures during discovery set ``status="pending"`` so that
``src/crawler/fetch.py`` can retry later.  URLs that cannot be fetched at
discovery time are still recorded (never silently dropped).

CLI::

    python -m src.crawler.discover_panopto \\
      --product panopto \\
      --out data/manifests/panopto_urls.jsonl \\
      [--max-pages 1000] \\
      [--delay-ms 1000] \\
      [--force]
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import click
import httpx
from bs4 import BeautifulSoup
from rich.console import Console

from .models import DiscoveryEntry, load_manifest, save_manifest

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_HOSTNAME = "support.panopto.com"
ROOT_SITEMAP_URL = "https://support.panopto.com/sitemap.xml"
ARTICLE_PREFIX = "/s/article/"
TOPIC_PREFIX = "/s/topic/"

DEFAULT_MAX_PAGES = 1000
DEFAULT_DELAY_MS = 1000

_DEFAULT_HEADERS = {
    "User-Agent": "rag-crawler/0.1 (educational internal use; contact lineusinc@gmail.com)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Known seed URLs to bootstrap BFS when sitemap is unavailable.
# These are canonical Panopto support article paths that are publicly documented.
SEED_URLS: list[str] = [
    "https://support.panopto.com/s/",
    "https://support.panopto.com/s/article/basics-of-panopto",
    "https://support.panopto.com/s/article/Recording-with-Panopto",
    "https://support.panopto.com/s/article/How-to-Create-a-Video-Assignment",
    "https://support.panopto.com/s/article/Editing-Your-Panopto-Recording",
    "https://support.panopto.com/s/article/Sharing-Your-Video",
    "https://support.panopto.com/s/article/How-to-Create-a-Folder",
    "https://support.panopto.com/s/article/How-to-Add-Users-to-a-Folder",
    "https://support.panopto.com/s/article/Canvas-Integration-Administrators",
    "https://support.panopto.com/s/article/Canvas-LTI-Student-Guide",
    "https://support.panopto.com/s/article/Captions-Overview",
    "https://support.panopto.com/s/article/Quizzes-in-Panopto",
    "https://support.panopto.com/s/article/Bookmarks-in-Panopto",
    "https://support.panopto.com/s/article/Recording-in-a-Classroom",
    "https://support.panopto.com/s/article/Panopto-for-Students",
    "https://support.panopto.com/s/article/Panopto-for-Instructors",
    "https://support.panopto.com/s/article/admin-guide",
]


# ---------------------------------------------------------------------------
# Pure URL helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip URL fragment and trailing slash for consistent deduplication."""
    try:
        p = urlparse(url)
        return p._replace(fragment="").geturl().rstrip("/")
    except Exception:
        return url


def is_panopto_url(url: str) -> bool:
    """Return True only for URLs on the allowed Panopto hostname."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.hostname == ALLOWED_HOSTNAME


def is_article_or_topic_url(url: str) -> bool:
    """Return True for /s/article/ or /s/topic/ URLs on support.panopto.com."""
    if not is_panopto_url(url):
        return False
    try:
        path = urlparse(url).path
    except Exception:
        return False
    return path.startswith(ARTICLE_PREFIX) or path.startswith(TOPIC_PREFIX)


def guide_from_url(url: str) -> str | None:
    """Return the last non-empty path segment (before any query string) as guide slug."""
    try:
        path = urlparse(url).path
        parts = [p for p in path.strip("/").split("/") if p]
    except Exception:
        return None
    if not parts:
        return None
    # Strip any trailing query artifacts from the last segment
    return parts[-1].split("?")[0] or None


def category_from_url(url: str) -> str | None:
    """
    Return the path segment immediately before the guide slug.

    For /s/article/some-slug  → 'article'
    For /s/topic/some-topic   → 'topic'
    """
    try:
        path = urlparse(url).path
        parts = [p for p in path.strip("/").split("/") if p]
    except Exception:
        return None
    if len(parts) >= 2:
        return parts[-2]
    return None


# ---------------------------------------------------------------------------
# Sitemap helpers (shared pattern with discover_zoom.py)
# ---------------------------------------------------------------------------

def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_sitemap(xml_text: str) -> list[str]:
    """Parse sitemap or sitemap-index XML; return all <loc> URLs."""
    root = ElementTree.fromstring(xml_text)
    urls: list[str] = []
    for el in root.iter():
        if _local_tag(el.tag) == "loc":
            url = (el.text or "").strip()
            if url:
                urls.append(url)
    return urls


def is_sitemap_index(xml_text: str) -> bool:
    """Return True if the XML document is a sitemap index (<sitemapindex>)."""
    try:
        root = ElementTree.fromstring(xml_text)
        return _local_tag(root.tag) == "sitemapindex"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# HTML parsing helpers (for BFS fallback)
# ---------------------------------------------------------------------------

def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all absolute links on the allowed hostname from raw HTML."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []
    seen: set[str] = set()
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"]).strip()
        if not href or href.startswith(("javascript:", "mailto:", "#", "tel:")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        try:
            p = urlparse(absolute)
        except Exception:
            continue
        if p.scheme not in ("http", "https"):
            continue
        if p.hostname != ALLOWED_HOSTNAME:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def extract_title(html: str) -> str | None:
    """Best-effort title extraction from raw HTML."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        val = str(og["content"]).strip()
        if val:
            return val
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text:
            return text
    title_tag = soup.find("title")
    if title_tag:
        text = title_tag.get_text(strip=True)
        if " | " in text:
            text = text.split(" | ")[0].strip()
        if text:
            return text
    return None


def extract_canonical(html: str) -> str | None:
    """Extract <link rel='canonical'> href if present."""
    try:
        soup = BeautifulSoup(html, "lxml")
        link = soup.find("link", rel="canonical")
        if link and link.get("href"):
            return str(link["href"]).strip() or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Sitemap-based discovery
# ---------------------------------------------------------------------------

def _discover_from_sitemap(
    sitemap_url: str,
    max_pages: int,
    delay_s: float,
    existing_urls: set[str],
    product: str,
    http_client: httpx.Client,
) -> tuple[list[DiscoveryEntry], bool]:
    """
    Attempt sitemap-based URL discovery.

    Returns (entries, success).  success=False means the sitemap was
    unreachable or contained no valid Panopto article URLs.
    """
    try:
        console.print(f"[blue]Trying sitemap: {sitemap_url}[/blue]")
        resp = http_client.get(sitemap_url)
        resp.raise_for_status()
        root_xml = resp.text
    except Exception as exc:
        console.print(f"[yellow]Sitemap fetch failed: {exc}[/yellow]")
        return [], False

    try:
        if is_sitemap_index(root_xml):
            child_urls = parse_sitemap(root_xml)
            console.print(f"[blue]Sitemap index with {len(child_urls)} child sitemap(s)[/blue]")
            all_locs: list[str] = []
            for i, child_url in enumerate(child_urls):
                if len(all_locs) >= max_pages * 2:
                    break
                if i > 0:
                    time.sleep(delay_s)
                try:
                    cr = http_client.get(child_url)
                    cr.raise_for_status()
                    all_locs.extend(parse_sitemap(cr.text))
                except Exception as exc:
                    console.print(f"[yellow]  WARN child sitemap {child_url}: {exc}[/yellow]")
        else:
            all_locs = parse_sitemap(root_xml)
            console.print(f"[blue]Sitemap contains {len(all_locs)} URL(s)[/blue]")
    except Exception as exc:
        console.print(f"[yellow]Sitemap parse failed: {exc}[/yellow]")
        return [], False

    entries: list[DiscoveryEntry] = []
    skipped = 0
    for raw_url in all_locs:
        if len(entries) >= max_pages:
            break
        url = normalize_url(raw_url)
        if not is_article_or_topic_url(url):
            skipped += 1
            continue
        if url in existing_urls:
            skipped += 1
            continue
        entries.append(DiscoveryEntry(
            source_url=url,
            canonical_url=url,
            title=None,
            product=product,
            guide=guide_from_url(url),
            category=category_from_url(url),
            role=None,
            discovered_at=datetime.now(timezone.utc),
            status="pending",
        ))

    console.print(
        f"[blue]Sitemap yielded {len(entries)} new article URLs ({skipped} skipped)[/blue]"
    )
    return entries, len(entries) > 0


# ---------------------------------------------------------------------------
# BFS-based discovery (fallback)
# ---------------------------------------------------------------------------

def _discover_bfs(
    seed_urls: list[str],
    max_pages: int,
    delay_s: float,
    existing_urls: set[str],
    product: str,
    http_client: httpx.Client,
    max_depth: int = 3,
) -> list[DiscoveryEntry]:
    """
    BFS from seed_urls collecting Panopto article/topic URLs.

    Fetch failures are tolerated: the seed URL itself is added as status='pending'
    so the main fetch stage can retry.  Never silently drops a known URL.
    """
    entries: list[DiscoveryEntry] = []
    visited: set[str] = set(existing_urls)
    # (url, depth)
    queue: deque[tuple[str, int]] = deque()

    for su in seed_urls:
        norm = normalize_url(su)
        if norm not in visited:
            queue.append((norm, 0))
            visited.add(norm)

    pages_fetched = 0

    while queue and len(entries) < max_pages:
        url, depth = queue.popleft()

        if pages_fetched > 0:
            time.sleep(delay_s)

        console.print(f"[dim][depth={depth}] {url}[/dim]")

        html: str | None = None
        try:
            resp = http_client.get(url)
            resp.raise_for_status()
            html = resp.text
            pages_fetched += 1
        except Exception as exc:
            console.print(f"[yellow]  WARN fetch failed: {exc}[/yellow]")
            # If URL itself is an article/topic, record it as pending (not as failed)
            if is_article_or_topic_url(url) and url not in {e.source_url for e in entries}:
                entries.append(DiscoveryEntry(
                    source_url=url,
                    canonical_url=url,
                    title=None,
                    product=product,
                    guide=guide_from_url(url),
                    category=category_from_url(url),
                    role=None,
                    discovered_at=datetime.now(timezone.utc),
                    status="pending",
                ))
            continue

        # Record the URL if it is an article/topic
        if is_article_or_topic_url(url) and url not in {e.source_url for e in entries}:
            title = extract_title(html) if html else None
            canonical = extract_canonical(html) if html else None
            entries.append(DiscoveryEntry(
                source_url=url,
                canonical_url=canonical or url,
                title=title,
                product=product,
                guide=guide_from_url(url),
                category=category_from_url(url),
                role=None,
                discovered_at=datetime.now(timezone.utc),
                status="pending",
            ))
            console.print(
                f"[green]  + [{len(entries)}] {title or url}[/green]"
            )

        # Follow links if depth allows
        if depth < max_depth and html:
            for link in extract_links(html, url):
                link_norm = normalize_url(link)
                if link_norm in visited:
                    continue
                # Only follow links within /s/ path scope
                try:
                    link_path = urlparse(link_norm).path
                except Exception:
                    continue
                if link_path.startswith("/s/"):
                    visited.add(link_norm)
                    queue.append((link_norm, depth + 1))

    return entries


# ---------------------------------------------------------------------------
# Public discovery entry point
# ---------------------------------------------------------------------------

def discover(
    sitemap_url: str = ROOT_SITEMAP_URL,
    seed_urls: list[str] | None = None,
    product: str = "panopto",
    max_pages: int = DEFAULT_MAX_PAGES,
    delay_s: float = DEFAULT_DELAY_MS / 1000.0,
    existing_urls: set[str] | None = None,
    http_client: httpx.Client | None = None,
) -> list[DiscoveryEntry]:
    """
    Discover Panopto support URLs via sitemap first, then BFS fallback.

    Parameters
    ----------
    sitemap_url:
        Root sitemap URL to attempt first.
    seed_urls:
        Override the built-in SEED_URLS for BFS fallback.
    product:
        Product name written into every DiscoveryEntry.
    max_pages:
        Upper bound on number of new entries to return.
    delay_s:
        Seconds to sleep between HTTP requests.
    existing_urls:
        Already-known URLs (for resume support).  These are skipped.
    http_client:
        Injected httpx.Client (for testing).  A new client is created if None.

    Returns
    -------
    list[DiscoveryEntry] with status='pending'.
    """
    own_client = http_client is None
    if own_client:
        http_client = httpx.Client(
            timeout=30.0,
            headers=_DEFAULT_HEADERS,
            follow_redirects=True,
        )

    if existing_urls is None:
        existing_urls = set()

    if seed_urls is None:
        seed_urls = SEED_URLS

    entries: list[DiscoveryEntry] = []

    try:
        # --- Strategy 1: sitemap ---
        sitemap_entries, sitemap_ok = _discover_from_sitemap(
            sitemap_url=sitemap_url,
            max_pages=max_pages,
            delay_s=delay_s,
            existing_urls=existing_urls,
            product=product,
            http_client=http_client,
        )

        if sitemap_ok:
            entries = sitemap_entries
        else:
            # --- Strategy 2: BFS from seeds ---
            console.print("[yellow]Falling back to BFS seed crawl[/yellow]")
            entries = _discover_bfs(
                seed_urls=seed_urls,
                max_pages=max_pages,
                delay_s=delay_s,
                existing_urls=existing_urls,
                product=product,
                http_client=http_client,
            )
    finally:
        if own_client:
            http_client.close()

    console.print(f"[bold]Total new Panopto URLs discovered: {len(entries)}[/bold]")
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--product",
    default="panopto",
    show_default=True,
    help="Product name written into every DiscoveryEntry",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output JSONL manifest path",
)
@click.option(
    "--max-pages",
    default=DEFAULT_MAX_PAGES,
    show_default=True,
    help="Maximum number of URLs to collect",
)
@click.option(
    "--delay-ms",
    default=DEFAULT_DELAY_MS,
    show_default=True,
    help="Delay between HTTP requests in milliseconds",
)
@click.option(
    "--force",
    is_flag=True,
    help="Ignore existing manifest and re-discover all URLs",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Discover without writing manifest to disk",
)
@click.option(
    "--sitemap-url",
    default=ROOT_SITEMAP_URL,
    show_default=True,
    help="Root sitemap XML URL",
)
def main(
    product: str,
    out: str,
    max_pages: int,
    delay_ms: int,
    force: bool,
    dry_run: bool,
    sitemap_url: str,
) -> None:
    """Discover Panopto support article URLs via sitemap or BFS fallback."""
    out_path = Path(out)

    if force:
        existing: list[DiscoveryEntry] = []
        existing_urls: set[str] = set()
        console.print("[yellow]--force: ignoring existing manifest[/yellow]")
    else:
        existing = load_manifest(out_path)
        existing_urls = {e.source_url for e in existing}
        if existing_urls:
            console.print(f"[blue]Resume: {len(existing_urls)} URLs already in manifest[/blue]")

    new_entries = discover(
        sitemap_url=sitemap_url,
        product=product,
        max_pages=max_pages,
        delay_s=delay_ms / 1000.0,
        existing_urls=existing_urls,
    )

    console.print(f"\n[bold]Discovered {len(new_entries)} new {product} URL(s)[/bold]")

    if dry_run:
        console.print("[yellow]--dry-run: manifest not written[/yellow]")
        for e in new_entries:
            console.print(f"  {e.source_url}")
        return

    all_entries = existing + new_entries
    save_manifest(all_entries, out_path)
    console.print(f"[green]Saved {len(all_entries)} total entries -> {out_path}[/green]")


if __name__ == "__main__":
    main()
