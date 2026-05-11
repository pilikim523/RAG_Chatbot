"""
Unit tests for src/crawler/discover_generic.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.crawler.discover_generic import (
    SKIP_EXTENSIONS,
    SKIP_PATH_PREFIXES,
    derive_category,
    derive_guide,
    discover,
    extract_links,
    extract_title,
    normalize_url,
    should_skip,
)
from src.crawler.models import DiscoveryEntry, load_manifest, save_manifest


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert normalize_url("https://example.com/docs#section") == "https://example.com/docs"

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/docs/") == "https://example.com/docs"

    def test_preserves_query_string(self):
        url = "https://example.com/docs?v=2"
        assert normalize_url(url) == url

    def test_handles_malformed(self):
        # Should not raise
        result = normalize_url("not_a_url")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------

class TestShouldSkip:
    HOSTNAME = "developers.zoom.us"

    def test_allows_valid_content_url(self):
        assert not should_skip(
            "https://developers.zoom.us/docs/api/meetings",
            self.HOSTNAME,
            url_prefix="/docs/",
        )

    def test_blocks_different_hostname(self):
        assert should_skip(
            "https://evil.com/docs/api",
            self.HOSTNAME,
            url_prefix=None,
        )

    def test_blocks_javascript_scheme(self):
        assert should_skip("javascript:void(0)", self.HOSTNAME, None)

    def test_blocks_asset_extension(self):
        for ext in [".jpg", ".css", ".js", ".pdf", ".mp4", ".woff2"]:
            url = f"https://{self.HOSTNAME}/static/file{ext}"
            assert should_skip(url, self.HOSTNAME, None), f"should skip {ext}"

    def test_blocks_utility_paths(self):
        for prefix in SKIP_PATH_PREFIXES:
            url = f"https://{self.HOSTNAME}{prefix}something"
            assert should_skip(url, self.HOSTNAME, None), f"should skip path {prefix}"

    def test_url_prefix_restriction(self):
        # URL outside prefix is skipped
        assert should_skip(
            "https://developers.zoom.us/blog/post-1",
            self.HOSTNAME,
            url_prefix="/docs/",
        )
        # URL inside prefix is allowed
        assert not should_skip(
            "https://developers.zoom.us/docs/api/users",
            self.HOSTNAME,
            url_prefix="/docs/",
        )

    def test_no_prefix_allows_all_paths(self):
        assert not should_skip(
            "https://developers.zoom.us/reference/list-meetings",
            self.HOSTNAME,
            url_prefix=None,
        )


# ---------------------------------------------------------------------------
# derive_guide and derive_category
# ---------------------------------------------------------------------------

class TestDeriveFunctions:
    def test_guide_last_segment(self):
        assert derive_guide("/docs/api/meetings") == "meetings"

    def test_guide_root(self):
        assert derive_guide("/") == ""

    def test_guide_single_segment(self):
        assert derive_guide("/docs") == "docs"

    def test_category_second_segment(self):
        assert derive_category("/docs/api/meetings") == "api"

    def test_category_single_segment(self):
        assert derive_category("/docs") == "docs"

    def test_category_root(self):
        assert derive_category("/") is None

    def test_category_two_segments(self):
        assert derive_category("/docs/api") == "api"


# ---------------------------------------------------------------------------
# extract_links
# ---------------------------------------------------------------------------

class TestExtractLinks:
    BASE = "https://developers.zoom.us/docs/"

    def test_finds_absolute_links(self):
        html = '<a href="https://developers.zoom.us/docs/api">API</a>'
        links = extract_links(html, self.BASE)
        assert "https://developers.zoom.us/docs/api" in links

    def test_finds_relative_links(self):
        html = '<a href="../reference">ref</a>'
        links = extract_links(html, self.BASE)
        assert any("reference" in l for l in links)

    def test_skips_javascript(self):
        html = '<a href="javascript:void(0)">bad</a>'
        assert extract_links(html, self.BASE) == []

    def test_skips_mailto(self):
        html = '<a href="mailto:test@example.com">mail</a>'
        assert extract_links(html, self.BASE) == []

    def test_deduplicates(self):
        html = '<a href="/docs/api">A</a><a href="/docs/api">B</a>'
        links = extract_links(html, self.BASE)
        assert links.count("https://developers.zoom.us/docs/api") == 1


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_og_title(self):
        html = '<meta property="og:title" content="Zoom API Reference">'
        assert extract_title(html) == "Zoom API Reference"

    def test_h1_fallback(self):
        html = "<h1>Getting Started</h1>"
        assert extract_title(html) == "Getting Started"

    def test_title_tag_strip_site_name(self):
        html = "<title>Create Meeting | Zoom Developer Docs</title>"
        assert extract_title(html) == "Create Meeting"

    def test_empty_html(self):
        assert extract_title("") is None


# ---------------------------------------------------------------------------
# discover (integration with mock HTTP)
# ---------------------------------------------------------------------------

def _make_html(title: str, links: list[str]) -> str:
    link_tags = "".join(f'<a href="{l}">{l}</a>' for l in links)
    return f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{link_tags}</body></html>"


class TestDiscover:
    HOSTNAME = "developers.zoom.us"
    START = "https://developers.zoom.us/docs"

    def _mock_client(self, pages: dict[str, str]) -> MagicMock:
        """Return a mock httpx.Client that serves pages dict."""
        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)

        def fake_get(url, **kwargs):
            resp = MagicMock()
            if url in pages:
                resp.text = pages[url]
                resp.raise_for_status = MagicMock()
                resp.status_code = 200
            else:
                resp.raise_for_status.side_effect = Exception("404 Not Found")
            return resp

        client.get = fake_get
        return client

    def test_records_start_page(self):
        pages = {
            self.START: _make_html("Zoom Docs", []),
        }
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                http_client=client,
            )
        assert len(entries) == 1
        assert entries[0].source_url == self.START
        assert entries[0].product == "zoom"
        assert entries[0].title == "Zoom Docs"
        assert entries[0].status == "pending"
        assert entries[0].role is None

    def test_follows_child_links(self):
        child = "https://developers.zoom.us/docs/api/meetings"
        pages = {
            self.START: _make_html("Zoom Docs", [child]),
            child: _make_html("Meetings API", []),
        }
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                http_client=client,
            )
        urls = [e.source_url for e in entries]
        assert self.START in urls
        assert child in urls

    def test_does_not_follow_external_links(self):
        external = "https://attacker.com/evil"
        pages = {
            self.START: _make_html("Zoom Docs", [external]),
        }
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                http_client=client,
            )
        urls = [e.source_url for e in entries]
        assert external not in urls

    def test_skips_existing_urls(self):
        pages = {
            self.START: _make_html("Zoom Docs", []),
        }
        client = self._mock_client(pages)
        existing = {self.START}
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                existing_urls=existing,
                http_client=client,
            )
        # Already visited, nothing new
        assert entries == []

    def test_url_prefix_filter(self):
        outside = "https://developers.zoom.us/blog/post"
        inside = "https://developers.zoom.us/docs/api/users"
        pages = {
            self.START: _make_html("Zoom Docs", [outside, inside]),
            inside: _make_html("Users API", []),
        }
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                url_prefix="/docs/",
                http_client=client,
            )
        urls = [e.source_url for e in entries]
        assert outside not in urls
        assert inside in urls

    def test_respects_max_pages(self):
        children = [f"https://developers.zoom.us/docs/page{i}" for i in range(10)]
        pages = {self.START: _make_html("Root", children)}
        for c in children:
            pages[c] = _make_html(f"Page {c[-1]}", [])
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                max_pages=3,
                http_client=client,
            )
        assert len(entries) <= 3

    def test_handles_fetch_error_gracefully(self):
        pages = {}  # No pages -> all requests raise 404
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url=self.START,
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                http_client=client,
            )
        assert entries == []

    def test_discovery_entry_fields(self):
        pages = {
            "https://developers.zoom.us/docs/api/meetings": _make_html("Meetings", []),
        }
        client = self._mock_client(pages)
        with patch("src.crawler.discover_generic.time.sleep"):
            entries = discover(
                start_url="https://developers.zoom.us/docs/api/meetings",
                product="zoom",
                allowed_hostname=self.HOSTNAME,
                http_client=client,
            )
        assert len(entries) == 1
        e = entries[0]
        assert e.guide == "meetings"
        assert e.category == "api"
        assert e.role is None
        assert isinstance(e.discovered_at, datetime)


# ---------------------------------------------------------------------------
# Manifest round-trip (uses models.py save/load)
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:
    def test_save_and_load(self, tmp_path: Path):
        entry = DiscoveryEntry(
            source_url="https://developers.zoom.us/docs/api/meetings",
            canonical_url="https://developers.zoom.us/docs/api/meetings",
            title="Meetings API",
            product="zoom",
            guide="meetings",
            category="api",
            role=None,
            discovered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            status="pending",
        )
        out = tmp_path / "zoom_urls.jsonl"
        save_manifest([entry], out)
        loaded = load_manifest(out)
        assert len(loaded) == 1
        assert loaded[0].source_url == entry.source_url
        assert loaded[0].product == "zoom"
        assert loaded[0].role is None
        assert loaded[0].status == "pending"
