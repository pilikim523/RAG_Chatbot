"""Tests for src/crawler/fetch.py."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from src.crawler.fetch import fetch_all, fetch_one, html_filename
from src.crawler.models import DiscoveryEntry, sha256_of

BASE = "https://community.instructure.com"
SAMPLE_HTML = "<html><body><h1>Canvas Guide</h1></body></html>"


def _pending(url: str) -> DiscoveryEntry:
    return DiscoveryEntry(
        source_url=url,
        discovered_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        status="pending",
    )


def _fetched(url: str) -> DiscoveryEntry:
    return DiscoveryEntry(
        source_url=url,
        discovered_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        status="fetched",
        raw_html_path=html_filename(url),
        content_hash=sha256_of(SAMPLE_HTML),
    )


GUIDE_URL = f"{BASE}/en/canvas/for-students/how-do-i-submit-an-assignment"


# ---------------------------------------------------------------------------
# html_filename
# ---------------------------------------------------------------------------

class TestHtmlFilename:
    def test_deterministic(self):
        assert html_filename(GUIDE_URL) == html_filename(GUIDE_URL)

    def test_ends_with_html(self):
        assert html_filename(GUIDE_URL).endswith(".html")

    def test_length(self):
        name = html_filename(GUIDE_URL)
        assert len(name) == 16 + len(".html")

    def test_different_urls_different_names(self):
        url2 = f"{BASE}/en/canvas/for-students/view-grades"
        assert html_filename(GUIDE_URL) != html_filename(url2)


# ---------------------------------------------------------------------------
# fetch_one
# ---------------------------------------------------------------------------

class TestFetchOne:
    @respx.mock
    async def test_success(self, tmp_path: Path):
        respx.get(GUIDE_URL).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))
        entry = _pending(GUIDE_URL)

        async with httpx.AsyncClient() as client:
            result = await fetch_one(entry, tmp_path, client, max_retries=1)

        assert result.status == "fetched"
        assert result.content_hash == sha256_of(SAMPLE_HTML)
        assert result.raw_html_path == html_filename(GUIDE_URL)
        saved_path = tmp_path / html_filename(GUIDE_URL)
        assert saved_path.exists()
        assert saved_path.read_text(encoding="utf-8") == SAMPLE_HTML

    @respx.mock
    async def test_404_marks_failed(self, tmp_path: Path):
        respx.get(GUIDE_URL).mock(return_value=httpx.Response(404))
        entry = _pending(GUIDE_URL)

        async with httpx.AsyncClient() as client:
            result = await fetch_one(entry, tmp_path, client, max_retries=0)

        assert result.status == "failed"
        assert "404" in (result.fetch_error or "")

    @respx.mock
    async def test_retries_on_500(self, tmp_path: Path):
        # Fail twice then succeed
        respx.get(GUIDE_URL).mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200, text=SAMPLE_HTML),
            ]
        )
        entry = _pending(GUIDE_URL)

        async with httpx.AsyncClient() as client:
            result = await fetch_one(entry, tmp_path, client, max_retries=3)

        assert result.status == "fetched"
        assert result.retry_count == 2

    @respx.mock
    async def test_exceeds_max_retries(self, tmp_path: Path):
        respx.get(GUIDE_URL).mock(return_value=httpx.Response(503))
        entry = _pending(GUIDE_URL)

        async with httpx.AsyncClient() as client:
            result = await fetch_one(entry, tmp_path, client, max_retries=2)

        assert result.status == "failed"

    @respx.mock
    async def test_network_error_retries(self, tmp_path: Path):
        respx.get(GUIDE_URL).mock(side_effect=httpx.ConnectError("timeout"))
        entry = _pending(GUIDE_URL)

        async with httpx.AsyncClient() as client:
            result = await fetch_one(entry, tmp_path, client, max_retries=1)

        assert result.status == "failed"
        assert result.fetch_error is not None

    @respx.mock
    async def test_content_hash_matches_html(self, tmp_path: Path):
        html = "<html><body>Specific content</body></html>"
        respx.get(GUIDE_URL).mock(return_value=httpx.Response(200, text=html))
        entry = _pending(GUIDE_URL)

        async with httpx.AsyncClient() as client:
            result = await fetch_one(entry, tmp_path, client, max_retries=0)

        assert result.content_hash == sha256_of(html)


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------

class TestFetchAll:
    @respx.mock
    async def test_fetches_all_pending(self, tmp_path: Path):
        urls = [f"{BASE}/en/canvas/for-students/guide-{i}" for i in range(3)]
        for url in urls:
            respx.get(url).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))

        entries = [_pending(u) for u in urls]
        results = await fetch_all(entries, tmp_path, concurrency=2, delay_s=0, max_retries=0)

        assert len(results) == 3
        fetched = [r for r in results if r.status == "fetched"]
        assert len(fetched) == 3

    @respx.mock
    async def test_skips_already_fetched(self, tmp_path: Path):
        url_pending = f"{BASE}/en/canvas/for-students/new-guide"
        url_done = f"{BASE}/en/canvas/for-students/old-guide"

        respx.get(url_pending).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))
        # url_done should NOT be fetched
        respx.get(url_done).mock(return_value=httpx.Response(500))

        # Create the already-fetched HTML file so status check passes
        (tmp_path / html_filename(url_done)).write_text(SAMPLE_HTML, encoding="utf-8")

        entries = [_pending(url_pending), _fetched(url_done)]
        results = await fetch_all(entries, tmp_path, concurrency=2, delay_s=0, max_retries=0)

        done_result = next(r for r in results if r.source_url == url_done)
        assert done_result.status == "fetched"

        new_result = next(r for r in results if r.source_url == url_pending)
        assert new_result.status == "fetched"

    @respx.mock
    async def test_partial_failure_continues(self, tmp_path: Path):
        url_ok = f"{BASE}/en/canvas/for-students/good-guide"
        url_bad = f"{BASE}/en/canvas/for-students/bad-guide"
        respx.get(url_ok).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))
        respx.get(url_bad).mock(return_value=httpx.Response(404))

        entries = [_pending(url_ok), _pending(url_bad)]
        results = await fetch_all(entries, tmp_path, concurrency=2, delay_s=0, max_retries=0)

        statuses = {r.source_url: r.status for r in results}
        assert statuses[url_ok] == "fetched"
        assert statuses[url_bad] == "failed"

    @respx.mock
    async def test_empty_pending_returns_unchanged(self, tmp_path: Path):
        url = f"{BASE}/en/canvas/for-students/done-guide"
        entries = [_fetched(url)]
        results = await fetch_all(entries, tmp_path, delay_s=0)
        assert len(results) == 1
        assert results[0].status == "fetched"

    @respx.mock
    async def test_creates_output_dir(self, tmp_path: Path):
        url = f"{BASE}/en/canvas/for-students/a-guide"
        respx.get(url).mock(return_value=httpx.Response(200, text=SAMPLE_HTML))

        out_dir = tmp_path / "deep" / "raw_html"
        entries = [_pending(url)]
        await fetch_all(entries, out_dir, delay_s=0, max_retries=0)

        assert out_dir.exists()
        assert (out_dir / html_filename(url)).exists()
