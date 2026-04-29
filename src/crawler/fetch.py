"""
HTML fetcher for Canvas guide URLs recorded in a discovery manifest.

Reads a JSONL manifest produced by discover.py, fetches pending entries,
saves raw HTML to out_dir, and writes an updated manifest with status,
content_hash, and raw_html_path filled in.

Supports:
  --concurrency   parallel fetchers (default 4)
  --delay-ms      minimum ms between consecutive requests (default 800)
  --max-retries   retries on 5xx / network error (default 3)
  --dry-run       print what would be fetched without saving anything
  --resume        (default on) skip entries already fetched successfully
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import click
import httpx
from rich.console import Console

from .models import DiscoveryEntry, load_manifest, save_manifest, sha256_of

console = Console(stderr=True)

_DEFAULT_HEADERS = {
    "User-Agent": "canvas-rag-crawler/0.1 (educational internal use; contact lineusinc@gmail.com)",
    "Accept-Language": "en-US,en;q=0.9",
}

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def html_filename(url: str) -> str:
    """Deterministic filename: first 16 hex chars of SHA-256(url)."""
    return f"{sha256_of(url)[:16]}.html"


# ---------------------------------------------------------------------------
# Single-URL fetch (injectable client for tests)
# ---------------------------------------------------------------------------

async def fetch_one(
    entry: DiscoveryEntry,
    out_dir: Path,
    client: httpx.AsyncClient,
    max_retries: int = 3,
) -> DiscoveryEntry:
    """Fetch one URL, save HTML, return updated DiscoveryEntry."""
    url = entry.source_url
    retry_count = 0

    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url)
        except httpx.RequestError as exc:
            if attempt < max_retries:
                retry_count += 1
                await asyncio.sleep(2 ** attempt)
                continue
            return entry.model_copy(update={
                "status": "failed",
                "fetch_error": f"RequestError: {exc}",
                "retry_count": retry_count,
            })

        if resp.status_code == 200:
            html = resp.text
            rel_path = html_filename(url)
            (out_dir / rel_path).write_text(html, encoding="utf-8")
            return entry.model_copy(update={
                "status": "fetched",
                "content_hash": sha256_of(html),
                "raw_html_path": rel_path,
                "fetch_error": None,
                "retry_count": retry_count,
            })

        if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
            retry_count += 1
            await asyncio.sleep(2 ** attempt)
            continue

        return entry.model_copy(update={
            "status": "failed",
            "fetch_error": f"HTTP {resp.status_code}",
            "retry_count": retry_count,
        })

    return entry.model_copy(update={
        "status": "failed",
        "fetch_error": "max retries exceeded",
        "retry_count": retry_count,
    })


# ---------------------------------------------------------------------------
# Concurrent fetch with rate limiting
# ---------------------------------------------------------------------------

async def fetch_all(
    entries: list[DiscoveryEntry],
    out_dir: Path,
    concurrency: int = 4,
    delay_s: float = 0.8,
    max_retries: int = 3,
) -> list[DiscoveryEntry]:
    """Fetch all pending entries; returns full updated list."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [e for e in entries if e.status in ("pending", "failed")]
    done = [e for e in entries if e.status not in ("pending", "failed")]

    if not pending:
        console.print("[dim]No pending entries to fetch[/dim]")
        return entries

    console.print(f"[blue]Fetching {len(pending)} URL(s) — concurrency={concurrency}, delay={delay_s:.2f}s[/blue]")

    # Worker: pulls from queue, fetches, sleeps delay_s between own requests
    results: list[DiscoveryEntry] = []
    lock = asyncio.Lock()
    queue: asyncio.Queue[DiscoveryEntry] = asyncio.Queue()
    for e in pending:
        queue.put_nowait(e)

    async def worker() -> None:
        first = True
        while True:
            try:
                entry = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not first:
                await asyncio.sleep(delay_s)
            first = False
            result = await fetch_one(entry, out_dir, client, max_retries)
            async with lock:
                results.append(result)
                idx = len(results)
                color = "green" if result.status == "fetched" else "red" if result.status == "failed" else "yellow"
                console.print(
                    f"[{color}][{idx}/{len(pending)}] {result.status}: {entry.source_url}[/{color}]"
                )

    async with httpx.AsyncClient(
        timeout=30.0,
        headers=_DEFAULT_HEADERS,
        follow_redirects=True,
    ) as client:
        workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(pending)))]
        await asyncio.gather(*workers)

    return done + results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Input JSONL manifest")
@click.option("--out-dir", required=True, type=click.Path(), help="Directory to save raw HTML")
@click.option("--concurrency", default=4, show_default=True, help="Parallel fetch workers")
@click.option("--delay-ms", default=800, show_default=True, help="Delay between requests per worker (ms)")
@click.option("--max-retries", default=3, show_default=True, help="Retry count on transient errors")
@click.option("--dry-run", is_flag=True, help="Print pending URLs without fetching")
def main(
    manifest: str,
    out_dir: str,
    concurrency: int,
    delay_ms: int,
    max_retries: int,
    dry_run: bool,
) -> None:
    """Fetch HTML for Canvas guide URLs in a discovery manifest."""
    manifest_path = Path(manifest)
    out_path = Path(out_dir)
    entries = load_manifest(manifest_path)

    if not entries:
        console.print("[yellow]Manifest is empty[/yellow]")
        return

    pending = [e for e in entries if e.status in ("pending", "failed")]
    console.print(f"[blue]{len(entries)} total entries, {len(pending)} pending[/blue]")

    if dry_run:
        console.print("[yellow]--dry-run: not fetching[/yellow]")
        for e in pending:
            console.print(f"  {e.source_url}")
        return

    updated = asyncio.run(
        fetch_all(
            entries=entries,
            out_dir=out_path,
            concurrency=concurrency,
            delay_s=delay_ms / 1000.0,
            max_retries=max_retries,
        )
    )

    save_manifest(updated, manifest_path)

    fetched = sum(1 for e in updated if e.status == "fetched")
    failed = sum(1 for e in updated if e.status == "failed")
    console.print(f"[bold]Done: {fetched} fetched, {failed} failed → {manifest_path}[/bold]")


if __name__ == "__main__":
    main()
