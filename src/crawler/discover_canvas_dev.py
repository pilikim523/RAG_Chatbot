"""
URL discovery for Instructure Developer Docs from developerdocs.instructure.com.

Strategy:
  1. Fetch https://developerdocs.instructure.com/sitemap-pages.xml
  2. Parse all <loc> entries
  3. Filter out index-only URLs (/, /get_started, /services)
  4. Derive category from /services/{service_name}/... pattern
  5. Emit DiscoveryEntry per URL (skip existing URLs for resume support)

robots.txt declares ai-train=yes; crawling is permitted.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import click
import httpx
from rich.console import Console

from .models import DiscoveryEntry, load_manifest, save_manifest

console = Console(stderr=True)

ALLOWED_HOSTNAME = "developerdocs.instructure.com"
SITEMAP_URL = "https://developerdocs.instructure.com/sitemap-pages.xml"

_DEFAULT_HEADERS = {
    "User-Agent": "canvas-rag-crawler/0.1 (educational internal use; contact lineusinc@gmail.com)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Index-only paths to exclude (not real content pages)
_INDEX_PATHS: set[str] = {"/", "/get_started", "/services"}

# Minimum path depth required to be a content page.
# /services/canvas         → depth 2 → excluded (section root, no article slug)
# /services/canvas/assignments → depth 3 → included
_MIN_CONTENT_DEPTH = 3


# ---------------------------------------------------------------------------
# Pure URL helpers
# ---------------------------------------------------------------------------

def is_index_url(path: str) -> bool:
    """
    Return True if path is an index-only page that should be skipped.

    Excluded:
    - Exact index paths: /, /get_started, /services
    - Service root pages with no article slug: /services/{name} (depth 2)

    Included (depth >= 3):
    - /services/{name}/{article}
    - /services/{name}/{sub}/{article}
    """
    normalized = "/" + path.strip("/")
    if not normalized.strip("/"):
        return True
    if normalized in _INDEX_PATHS:
        return True
    # Exclude bare service section roots: /services/{name} with no further slug
    parts = [p for p in normalized.strip("/").split("/") if p]
    if len(parts) < _MIN_CONTENT_DEPTH:
        return True
    return False


def category_from_path(path: str) -> str | None:
    """
    Extract service category from URL path.

    /services/canvas/...         → 'canvas'
    /services/dap/...            → 'dap'
    /get_started                 → None
    """
    parts = [p for p in path.strip("/").split("/") if p]
    # /services/{service_name}/... → parts[0]='services', parts[1]=service_name
    if len(parts) >= 2 and parts[0] == "services":
        return parts[1]
    return None


def guide_from_path(path: str) -> str | None:
    """Return the last non-empty path segment as the guide slug."""
    parts = [p for p in path.strip("/").split("/") if p]
    if parts:
        return parts[-1]
    return None


def is_developer_url(url: str) -> bool:
    """Return True only for URLs on the allowed developer docs hostname."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.hostname == ALLOWED_HOSTNAME


def normalize_url(url: str) -> str:
    """Strip URL fragment and trailing slash."""
    try:
        p = urlparse(url)
        return p._replace(fragment="").geturl().rstrip("/")
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------

def parse_sitemap(xml_text: str) -> list[str]:
    """
    Parse sitemap XML and return list of <loc> URLs.

    Handles both plain sitemaps and sitemap index files.
    Namespace-agnostic: strips namespace prefix before comparing tags.
    """
    root = ElementTree.fromstring(xml_text)

    def _local(tag: str) -> str:
        """Strip XML namespace from tag name."""
        return tag.split("}")[-1] if "}" in tag else tag

    urls: list[str] = []
    for el in root.iter():
        if _local(el.tag) == "loc":
            url = (el.text or "").strip()
            if url:
                urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# Core discovery (sitemap-based)
# ---------------------------------------------------------------------------

def discover_from_sitemap(
    sitemap_url: str = SITEMAP_URL,
    delay_s: float = 0.8,
    existing_urls: set[str] | None = None,
    http_client: httpx.Client | None = None,
) -> list[DiscoveryEntry]:
    """
    Fetch sitemap XML and build DiscoveryEntry list.

    Skips:
    - URLs not on ALLOWED_HOSTNAME
    - Index-only paths (/, /services, /get_started)
    - URLs already present in existing_urls (resume support)

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

    try:
        console.print(f"[blue]Fetching sitemap: {sitemap_url}[/blue]")
        resp = http_client.get(sitemap_url)
        resp.raise_for_status()
        xml_text = resp.text
    finally:
        if own_client:
            http_client.close()

    raw_urls = parse_sitemap(xml_text)
    console.print(f"[blue]Sitemap contains {len(raw_urls)} URL(s)[/blue]")

    entries: list[DiscoveryEntry] = []
    skipped_count = 0

    for raw_url in raw_urls:
        url = normalize_url(raw_url)

        # Enforce hostname
        if not is_developer_url(url):
            skipped_count += 1
            continue

        path = urlparse(url).path

        # Skip index-only pages
        if is_index_url(path):
            skipped_count += 1
            continue

        # Resume: skip already-known URLs
        if url in existing_urls:
            skipped_count += 1
            continue

        category = category_from_path(path)
        guide = guide_from_path(path)

        entry = DiscoveryEntry(
            source_url=url,
            canonical_url=url,
            title=None,
            product="developer",
            guide=guide,
            category=category,
            role=None,
            discovered_at=datetime.now(timezone.utc),
            status="pending",
        )
        entries.append(entry)

    console.print(
        f"[bold]Found {len(entries)} new URL(s) ({skipped_count} skipped)[/bold]"
    )
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--sitemap-url",
    default=SITEMAP_URL,
    show_default=True,
    help="Sitemap XML URL to parse",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output JSONL manifest path",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Discover without writing manifest to disk",
)
@click.option(
    "--delay-ms",
    default=800,
    show_default=True,
    help="Delay between HTTP requests (ms) — reserved for future multi-request use",
)
def main(
    sitemap_url: str,
    out: str,
    dry_run: bool,
    delay_ms: int,
) -> None:
    """Discover Instructure Developer Docs URLs from sitemap-pages.xml."""
    out_path = Path(out)
    existing = load_manifest(out_path)
    existing_urls = {e.source_url for e in existing}
    if existing_urls:
        console.print(f"[blue]Resume: {len(existing_urls)} URLs already in manifest[/blue]")

    new_entries = discover_from_sitemap(
        sitemap_url=sitemap_url,
        delay_s=delay_ms / 1000.0,
        existing_urls=existing_urls,
    )

    console.print(f"\n[bold]Discovered {len(new_entries)} new developer doc URL(s)[/bold]")

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
