"""
Generic BFS URL discovery crawler for any documentation website.

Restricts crawling to the start_url's hostname (or an explicit --allowed-hostname).
All visited content pages are recorded as DiscoveryEntry with status='pending'.

Typical usage (Zoom developer docs):

    uv run python -m src.crawler.discover_generic \\
      --start-url "https://developers.zoom.us/docs/" \\
      --product zoom \\
      --url-prefix /docs/ \\
      --out data/manifests/zoom_urls.jsonl

BFS strategy:
  - Enqueue start_url at depth=0
  - Follow same-host links within optional url-prefix up to max-depth
  - Skip asset / utility URLs (see SKIP_EXTENSIONS, SKIP_PATH_PREFIXES)
  - Every successfully fetched page becomes a DiscoveryEntry
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

from .models import DiscoveryEntry, load_manifest, save_manifest

console = Console(stderr=True)

_USER_AGENT = (
    "rag-crawler/0.1 (educational internal use; contact lineusinc@gmail.com)"
)

_DEFAULT_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# File extensions to skip (assets, data files, media)
SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
        ".css", ".js", ".json", ".xml", ".map",
        ".pdf", ".zip", ".tar", ".gz",
        ".mp4", ".mp3",
        ".woff", ".woff2", ".ttf", ".eot",
    }
)

# Path prefixes that indicate non-content pages
SKIP_PATH_PREFIXES: tuple[str, ...] = (
    "/cdn-cgi/",
    "/login",
    "/logout",
    "/signup",
    "/oauth/",
    "/auth/",
    "/static/",
    "/assets/",
    "/__",
)


# ---------------------------------------------------------------------------
# Pure URL helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip fragment and trailing slash for consistent deduplication."""
    try:
        p = urlparse(url)
        return p._replace(fragment="").geturl().rstrip("/")
    except Exception:
        return url


def should_skip(url: str, allowed_hostname: str, url_prefix: str | None) -> bool:
    """Return True if this URL should not be crawled."""
    try:
        p = urlparse(url)
    except Exception:
        return True

    if p.scheme not in ("http", "https"):
        return True

    if p.hostname != allowed_hostname:
        return True

    path = p.path

    # Path prefix restriction: allow the prefix itself (with or without trailing slash)
    if url_prefix and not path.startswith(url_prefix):
        # Also allow the bare prefix path (e.g. /docs matches /docs/)
        bare_prefix = url_prefix.rstrip("/")
        if path != bare_prefix:
            return True

    # Skip utility paths
    for prefix in SKIP_PATH_PREFIXES:
        if path.startswith(prefix):
            return True

    # Skip asset extensions
    path_lower = path.lower().split("?")[0]
    ext = "." + path_lower.rsplit(".", 1)[-1] if "." in path_lower.rsplit("/", 1)[-1] else ""
    if ext in SKIP_EXTENSIONS:
        return True

    return False


def derive_guide(path: str) -> str:
    """Last non-empty path segment."""
    parts = [p for p in path.rstrip("/").split("/") if p]
    return parts[-1] if parts else ""


def derive_category(path: str) -> str | None:
    """Second path segment (e.g. /docs/api/meetings -> 'api')."""
    parts = [p for p in path.strip("/").split("/") if p]
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else None)


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
    product: str,
    allowed_hostname: str,
    url_prefix: str | None = None,
    max_depth: int = 8,
    max_pages: int = 3000,
    delay_s: float = 1.0,
    existing_urls: set[str] | None = None,
    http_client: httpx.Client | None = None,
) -> list[DiscoveryEntry]:
    """
    BFS from start_url collecting documentation pages.

    Every successfully fetched content page becomes a DiscoveryEntry.
    Returns entries with status='pending'.
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

    start_norm = normalize_url(start_url)
    # (url, depth)
    queue: deque[tuple[str, int]] = deque([(start_norm, 0)])
    pages_fetched = 0

    try:
        while queue and len(entries) < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if should_skip(url, allowed_hostname, url_prefix):
                continue

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

            parsed = urlparse(url)
            entry = DiscoveryEntry(
                source_url=url,
                canonical_url=extract_canonical(html) or url,
                title=extract_title(html),
                product=product,
                guide=derive_guide(parsed.path),
                category=derive_category(parsed.path),
                role=None,
                discovered_at=datetime.now(timezone.utc),
                status="pending",
            )
            entries.append(entry)
            console.print(
                f"[green]  + [{len(entries)}] {entry.title or url}[/green]"
            )

            # Do not follow links beyond max_depth
            if depth >= max_depth:
                continue

            for link in extract_links(html, url):
                link_norm = normalize_url(link)
                if link_norm in visited:
                    continue
                if should_skip(link_norm, allowed_hostname, url_prefix):
                    continue
                queue.append((link_norm, depth + 1))

    finally:
        if own_client:
            http_client.close()

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--start-url", required=True, help="URL to start BFS discovery from")
@click.option("--product", required=True, help="Product name (e.g. zoom, confluence)")
@click.option("--out", required=True, type=click.Path(), help="Output JSONL manifest path")
@click.option("--dry-run", is_flag=True, help="Discover without writing manifest to disk")
@click.option("--max-depth", default=8, show_default=True, help="Max BFS depth")
@click.option("--max-pages", default=3000, show_default=True, help="Max pages to collect")
@click.option("--delay-ms", default=1000, show_default=True, help="Delay between requests (ms)")
@click.option(
    "--allowed-hostname",
    default=None,
    help="Override allowed hostname (defaults to start-url's hostname)",
)
@click.option(
    "--url-prefix",
    default=None,
    help="Restrict crawl to URLs whose path starts with this prefix (e.g. /docs/)",
)
def main(
    start_url: str,
    product: str,
    out: str,
    dry_run: bool,
    max_depth: int,
    max_pages: int,
    delay_ms: int,
    allowed_hostname: str | None,
    url_prefix: str | None,
) -> None:
    """Generic BFS documentation crawler. Restricted to start-url's hostname."""
    parsed_start = urlparse(start_url)
    hostname = allowed_hostname or parsed_start.hostname
    if not hostname:
        raise click.ClickException(f"Cannot determine hostname from start-url: {start_url}")

    console.print(f"[blue]Crawling [bold]{hostname}[/bold] | product=[bold]{product}[/bold][/blue]")
    if url_prefix:
        console.print(f"[blue]URL prefix restriction: {url_prefix}[/blue]")

    out_path = Path(out)
    existing = load_manifest(out_path)
    existing_urls = {e.source_url for e in existing}
    if existing_urls:
        console.print(f"[blue]Resume: {len(existing_urls)} URLs already in manifest[/blue]")

    new_entries = discover(
        start_url=start_url,
        product=product,
        allowed_hostname=hostname,
        url_prefix=url_prefix,
        max_depth=max_depth,
        max_pages=max_pages,
        delay_s=delay_ms / 1000.0,
        existing_urls=existing_urls,
    )

    console.print(f"\n[bold]Discovered {len(new_entries)} new {product} page(s)[/bold]")

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
