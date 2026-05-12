"""
URL discovery for Zoom Developer Docs from developers.zoom.us.

Zoom Developer Docs is a Next.js SSG site; sitemap-based URL collection
is more reliable than BFS HTML crawling because the JS-rendered nav is
not present in raw HTML responses.

Strategy:
  1. Fetch https://developers.zoom.us/sitemap.xml
  2. If the root sitemap is a sitemap index, fetch each child sitemap.
  3. Parse all <loc> entries from every sitemap.
  4. Filter to URLs whose path starts with --url-prefix (default: /docs/).
  5. Emit one DiscoveryEntry per URL; skip existing URLs for resume support.

CLI:
    python -m src.crawler.discover_zoom \\
        --out data/manifests/zoom_urls.jsonl \\
        --product zoom \\
        --url-prefix /docs/ \\
        --max-pages 2000

Resume: if --out already exists, URLs already in the manifest are skipped.
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

ALLOWED_HOSTNAME = "developers.zoom.us"
ROOT_SITEMAP_URL = "https://developers.zoom.us/sitemap.xml"
DEFAULT_URL_PREFIX = "/docs/"
DEFAULT_MAX_PAGES = 2000
DEFAULT_DELAY_MS = 500

_DEFAULT_HEADERS = {
    "User-Agent": "rag-crawler/0.1 (educational internal use; contact lineusinc@gmail.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


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


def is_zoom_url(url: str) -> bool:
    """Return True only for URLs on the allowed Zoom hostname."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.hostname == ALLOWED_HOSTNAME


def has_url_prefix(url: str, prefix: str) -> bool:
    """Return True if url's path starts with prefix."""
    try:
        path = urlparse(url).path
    except Exception:
        return False
    return path.startswith(prefix)


def guide_from_url(url: str) -> str | None:
    """Return the last non-empty path segment as the guide slug."""
    try:
        parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    except Exception:
        return None
    return parts[-1] if parts else None


def category_from_url(url: str, url_prefix: str = DEFAULT_URL_PREFIX) -> str | None:
    """
    Return the second path segment relative to the url_prefix.

    For /docs/api-reference/zoom-meetings/... the prefix is /docs/ so
    path segments under the prefix are ['api-reference', 'zoom-meetings', ...].
    The category is the first such segment ('api-reference').
    """
    try:
        path = urlparse(url).path
    except Exception:
        return None

    stripped_prefix = url_prefix.rstrip("/")
    if path.startswith(stripped_prefix):
        remainder = path[len(stripped_prefix):]
    elif path.startswith(url_prefix):
        remainder = path[len(url_prefix):]
    else:
        remainder = path

    parts = [p for p in remainder.strip("/").split("/") if p]
    return parts[0] if parts else None


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------

def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from tag name."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_sitemap(xml_text: str) -> list[str]:
    """
    Parse sitemap or sitemap-index XML; return all <loc> URLs.

    Works for:
    - Plain sitemaps (<urlset><url><loc>...</loc></url></urlset>)
    - Sitemap index files (<sitemapindex><sitemap><loc>...</loc></sitemap>)
    - Both with and without XML namespaces
    """
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
# Core discovery (sitemap-based)
# ---------------------------------------------------------------------------

def discover(
    sitemap_url: str = ROOT_SITEMAP_URL,
    url_prefix: str = DEFAULT_URL_PREFIX,
    max_pages: int = DEFAULT_MAX_PAGES,
    delay_s: float = DEFAULT_DELAY_MS / 1000.0,
    existing_urls: set[str] | None = None,
    http_client: httpx.Client | None = None,
) -> list[DiscoveryEntry]:
    """
    Fetch Zoom sitemap(s) and build a DiscoveryEntry list.

    Handles sitemap index files by fetching each child sitemap.
    Filters to URLs on ALLOWED_HOSTNAME whose path starts with url_prefix.
    Skips URLs already present in existing_urls (resume support).

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
    skipped_count = 0

    try:
        # Step 1: fetch the root sitemap
        console.print(f"[blue]Fetching root sitemap: {sitemap_url}[/blue]")
        resp = http_client.get(sitemap_url)
        resp.raise_for_status()
        root_xml = resp.text

        # Step 2: collect child sitemap URLs or direct page URLs
        if is_sitemap_index(root_xml):
            child_sitemap_urls = parse_sitemap(root_xml)
            console.print(
                f"[blue]Sitemap index: {len(child_sitemap_urls)} child sitemap(s)[/blue]"
            )
            all_page_urls: list[str] = []
            for i, child_url in enumerate(child_sitemap_urls):
                if len(all_page_urls) >= max_pages:
                    break
                if i > 0:
                    time.sleep(delay_s)
                console.print(f"[dim]  Fetching child sitemap ({i + 1}/{len(child_sitemap_urls)}): {child_url}[/dim]")
                try:
                    child_resp = http_client.get(child_url)
                    child_resp.raise_for_status()
                    all_page_urls.extend(parse_sitemap(child_resp.text))
                except Exception as exc:
                    console.print(f"[yellow]  WARN child sitemap {child_url}: {exc}[/yellow]")
        else:
            all_page_urls = parse_sitemap(root_xml)
            console.print(f"[blue]Sitemap contains {len(all_page_urls)} URL(s)[/blue]")

        # Step 3: filter and build entries
        for raw_url in all_page_urls:
            if len(entries) >= max_pages:
                console.print(f"[yellow]  Reached max_pages={max_pages}; stopping.[/yellow]")
                break

            url = normalize_url(raw_url)

            # Enforce hostname
            if not is_zoom_url(url):
                skipped_count += 1
                continue

            # Enforce path prefix
            if not has_url_prefix(url, url_prefix):
                skipped_count += 1
                continue

            # Resume: skip already-known URLs
            if url in existing_urls:
                skipped_count += 1
                continue

            entry = DiscoveryEntry(
                source_url=url,
                canonical_url=url,
                title=None,
                product="zoom",
                guide=guide_from_url(url),
                category=category_from_url(url, url_prefix),
                role=None,
                discovered_at=datetime.now(timezone.utc),
                status="pending",
            )
            entries.append(entry)

    finally:
        if own_client:
            http_client.close()

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
    default=ROOT_SITEMAP_URL,
    show_default=True,
    help="Root sitemap XML URL (may be a sitemap index)",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output JSONL manifest path",
)
@click.option(
    "--product",
    default="zoom",
    show_default=True,
    help="Product name written into every DiscoveryEntry",
)
@click.option(
    "--url-prefix",
    default=DEFAULT_URL_PREFIX,
    show_default=True,
    help="Only collect URLs whose path starts with this prefix",
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
    "--dry-run",
    is_flag=True,
    help="Discover without writing manifest to disk",
)
def main(
    sitemap_url: str,
    out: str,
    product: str,
    url_prefix: str,
    max_pages: int,
    delay_ms: int,
    dry_run: bool,
) -> None:
    """Discover Zoom Developer Docs URLs from sitemap.xml."""
    out_path = Path(out)
    existing = load_manifest(out_path)
    existing_urls = {e.source_url for e in existing}
    if existing_urls:
        console.print(f"[blue]Resume: {len(existing_urls)} URLs already in manifest[/blue]")

    new_entries = discover(
        sitemap_url=sitemap_url,
        url_prefix=url_prefix,
        max_pages=max_pages,
        delay_s=delay_ms / 1000.0,
        existing_urls=existing_urls,
    )
    # Override product field if caller passed a different value
    if product != "zoom":
        for e in new_entries:
            e.product = product

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
