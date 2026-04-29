"""
URL discovery for Canvas guides from community.instructure.com/en/all-guides.

Real URL structure (discovered from the live site):
  /en/all-guides
    → /en/kb/canvas-lms-student-guide       (collection index, 3 segments)
    → /en/kb/canvas-lms-instructor-guide    (collection index, 3 segments)
    → ...
      → /en/kb/articles/{id}-{slug}         (individual article, 4 segments)

BFS strategy:
  depth=0  (start URL)     → queue canvas collection URLs
  depth=1  (collection)    → queue article URLs, record role/category from collection slug
  depth≥2  (articles)      → leaf nodes, added to manifest, not followed
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import click
import httpx
from bs4 import BeautifulSoup
from rich.console import Console

from .models import (
    CANVAS_COLLECTIONS,
    DiscoveryEntry,
    collection_metadata,
    load_manifest,
    save_manifest,
)

console = Console(stderr=True)

ALLOWED_HOSTNAME = "community.instructure.com"

_DEFAULT_HEADERS = {
    "User-Agent": "canvas-rag-crawler/0.1 (educational internal use; contact lineusinc@gmail.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Pure URL filter helpers
# ---------------------------------------------------------------------------

def is_canvas_collection_url(url: str) -> bool:
    """Return True for /en/kb/<canvas-collection-slug> pages."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.hostname != ALLOWED_HOSTNAME:
        return False
    path = p.path.rstrip("/")
    if not path.startswith("/en/kb/"):
        return False
    slug = path[len("/en/kb/"):].split("/")[0].split("?")[0]
    return slug in CANVAS_COLLECTIONS


def is_article_url(url: str) -> bool:
    """Return True for /en/kb/articles/... individual article pages."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.hostname == ALLOWED_HOSTNAME and p.path.startswith("/en/kb/articles/")


def is_guide_url(url: str) -> bool:
    """Return True for URLs that are individual guide articles (4+ path segments)."""
    return is_article_url(url)


def collection_slug_from_url(url: str) -> str | None:
    """Extract collection slug from a canvas collection URL."""
    try:
        path = urlparse(url).path.rstrip("/")
    except Exception:
        return None
    if path.startswith("/en/kb/"):
        slug = path[len("/en/kb/"):].split("/")[0].split("?")[0]
        if slug in CANVAS_COLLECTIONS:
            return slug
    return None


def normalize_url(url: str) -> str:
    """Strip URL fragment and trailing slash."""
    try:
        p = urlparse(url)
        return p._replace(fragment="").geturl().rstrip("/")
    except Exception:
        return url


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
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
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return str(og["content"]).strip() or None
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
    soup = BeautifulSoup(html, "lxml")
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return str(link["href"]).strip() or None
    return None


# ---------------------------------------------------------------------------
# Core BFS discovery
# ---------------------------------------------------------------------------

def discover(
    start_url: str,
    product: str = "canvas",
    max_depth: int = 3,
    max_pages: int = 500,
    delay_s: float = 0.8,
    existing_urls: set[str] | None = None,
    http_client: httpx.Client | None = None,
) -> list[DiscoveryEntry]:
    """
    BFS from start_url collecting Canvas guide articles.

    Queue items: (url, depth, parent_collection_slug | None)
    Returns DiscoveryEntry list with status='pending'.
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

    entries: list[DiscoveryEntry] = []
    visited: set[str] = set(existing_urls)

    # If start_url is already a collection, begin at depth=1 so that
    # article links on that page are queued (depth=0 only follows collections).
    start_norm = normalize_url(start_url)
    start_slug = collection_slug_from_url(start_norm)
    initial_item: tuple[str, int, str | None] = (
        (start_norm, 1, start_slug) if start_slug else (start_norm, 0, None)
    )

    # (url, depth, parent_collection_slug)
    queue: deque[tuple[str, int, str | None]] = deque([initial_item])
    pages_fetched = 0

    try:
        while queue and len(entries) < max_pages:
            url, depth, parent_collection = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if pages_fetched > 0:
                time.sleep(delay_s)

            console.print(f"[dim][depth={depth}] {url}[/dim]")
            try:
                resp = http_client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                console.print(f"[yellow]  WARN: {exc}[/yellow]")
                continue

            pages_fetched += 1
            html = resp.text

            if is_article_url(url) and parent_collection is not None:
                # Leaf: individual Canvas guide article
                role, sub_product = collection_metadata(parent_collection)
                entry = DiscoveryEntry(
                    source_url=url,
                    canonical_url=extract_canonical(html) or url,
                    title=extract_title(html),
                    product=product,
                    guide=url.rstrip("/").split("/")[-1],
                    category=parent_collection,
                    role=role,
                    discovered_at=datetime.now(timezone.utc),
                    status="pending",
                )
                entries.append(entry)
                console.print(
                    f"[green]  ✓ [{len(entries)}] {entry.title or url}[/green]"
                )
                # Articles are leaf nodes – do not follow their links
                continue

            if depth >= max_depth:
                continue

            # Index/collection page: queue relevant child links
            for link in extract_links(html, url):
                link_norm = normalize_url(link)
                if link_norm in visited:
                    continue

                if depth == 0:
                    # From all-guides (or custom start): only follow canvas collections
                    slug = collection_slug_from_url(link_norm)
                    if slug:
                        queue.append((link_norm, depth + 1, slug))
                else:
                    # From a canvas collection: queue article links
                    if is_article_url(link_norm):
                        queue.append((link_norm, depth + 1, parent_collection))
                    # Also follow pagination on collection pages (same collection)
                    elif is_canvas_collection_url(link_norm):
                        slug = collection_slug_from_url(link_norm)
                        if slug:
                            queue.append((link_norm, depth + 1, slug))

    finally:
        if own_client:
            http_client.close()

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start-url", required=True, help="URL to start discovery from")
@click.option("--product", default="canvas", show_default=True, help="Product filter")
@click.option("--out", required=True, type=click.Path(), help="Output JSONL manifest path")
@click.option("--dry-run", is_flag=True, help="Discover without writing manifest to disk")
@click.option("--max-depth", default=3, show_default=True, help="Max BFS depth")
@click.option("--max-pages", default=500, show_default=True, help="Max guide articles to collect")
@click.option("--delay-ms", default=800, show_default=True, help="Delay between requests (ms)")
def main(
    start_url: str,
    product: str,
    out: str,
    dry_run: bool,
    max_depth: int,
    max_pages: int,
    delay_ms: int,
) -> None:
    """Discover Canvas guide URLs from the Instructure Community."""
    out_path = Path(out)
    existing = load_manifest(out_path)
    existing_urls = {e.source_url for e in existing}
    if existing_urls:
        console.print(f"[blue]Resume: {len(existing_urls)} URLs already in manifest[/blue]")

    new_entries = discover(
        start_url=start_url,
        product=product,
        max_depth=max_depth,
        max_pages=max_pages,
        delay_s=delay_ms / 1000.0,
        existing_urls=existing_urls,
    )

    console.print(f"\n[bold]Discovered {len(new_entries)} new {product} guide URL(s)[/bold]")

    if dry_run:
        console.print("[yellow]--dry-run: manifest not written[/yellow]")
        for e in new_entries:
            console.print(f"  {e.source_url}")
        return

    all_entries = existing + new_entries
    save_manifest(all_entries, out_path)
    console.print(f"[green]Saved {len(all_entries)} total entries → {out_path}[/green]")


if __name__ == "__main__":
    main()
