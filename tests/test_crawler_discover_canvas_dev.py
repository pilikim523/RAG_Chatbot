"""
Tests for src/crawler/discover_canvas_dev.py.

All HTTP calls are intercepted with respx so no network access is required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from src.crawler.discover_canvas_dev import (
    ALLOWED_HOSTNAME,
    SITEMAP_URL,
    category_from_path,
    discover_from_sitemap,
    guide_from_path,
    is_developer_url,
    is_index_url,
    normalize_url,
    parse_sitemap,
)
from src.crawler.models import DiscoveryEntry, save_manifest

# ---------------------------------------------------------------------------
# Sample sitemap XML fixture
# ---------------------------------------------------------------------------

SITEMAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://developerdocs.instructure.com/</loc></url>
  <url><loc>https://developerdocs.instructure.com/get_started</loc></url>
  <url><loc>https://developerdocs.instructure.com/services</loc></url>
  <url><loc>https://developerdocs.instructure.com/services/canvas</loc></url>
  <url><loc>https://developerdocs.instructure.com/services/canvas/assignments</loc></url>
  <url><loc>https://developerdocs.instructure.com/services/canvas/courses</loc></url>
  <url><loc>https://developerdocs.instructure.com/services/dap/overview</loc></url>
  <url><loc>https://developerdocs.instructure.com/services/studio/getting-started</loc></url>
  <url><loc>https://other.example.com/services/canvas/page</loc></url>
</urlset>
"""

# Minimal sitemap without XML namespace (still valid)
SITEMAP_XML_NO_NS = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url><loc>https://developerdocs.instructure.com/services/canvas/assignments</loc></url>
  <url><loc>https://developerdocs.instructure.com/services/dap/overview</loc></url>
</urlset>
"""


# ---------------------------------------------------------------------------
# parse_sitemap
# ---------------------------------------------------------------------------

class TestParseSitemap:
    def test_returns_all_loc_urls(self):
        urls = parse_sitemap(SITEMAP_XML)
        assert "https://developerdocs.instructure.com/services/canvas/assignments" in urls
        assert "https://developerdocs.instructure.com/services/dap/overview" in urls
        assert "https://other.example.com/services/canvas/page" in urls

    def test_total_count(self):
        urls = parse_sitemap(SITEMAP_XML)
        assert len(urls) == 9

    def test_no_namespace_sitemap(self):
        urls = parse_sitemap(SITEMAP_XML_NO_NS)
        assert len(urls) == 2
        assert "https://developerdocs.instructure.com/services/canvas/assignments" in urls


# ---------------------------------------------------------------------------
# is_index_url
# ---------------------------------------------------------------------------

class TestIsIndexUrl:
    @pytest.mark.parametrize("path", ["/", "/services", "/get_started", ""])
    def test_index_paths_are_excluded(self, path: str):
        assert is_index_url(path) is True

    @pytest.mark.parametrize("path", [
        "/services/canvas/assignments",
        "/services/dap/overview",
        "/services/ab-connect/endpoints",
    ])
    def test_content_paths_are_not_index(self, path: str):
        assert is_index_url(path) is False

    def test_bare_service_root_is_index(self):
        # /services/canvas with no article slug is a section root, not content
        assert is_index_url("/services/canvas") is True


# ---------------------------------------------------------------------------
# category_from_path
# ---------------------------------------------------------------------------

class TestCategoryFromPath:
    @pytest.mark.parametrize("path,expected", [
        ("/services/canvas/assignments", "canvas"),
        ("/services/dap/overview", "dap"),
        ("/services/ab-connect/endpoints", "ab-connect"),
        ("/services/studio/getting-started", "studio"),
        ("/services/canvas", "canvas"),
    ])
    def test_extracts_service_name(self, path: str, expected: str):
        assert category_from_path(path) == expected

    @pytest.mark.parametrize("path", ["/get_started", "/", "/about"])
    def test_non_service_paths_return_none(self, path: str):
        assert category_from_path(path) is None


# ---------------------------------------------------------------------------
# guide_from_path
# ---------------------------------------------------------------------------

class TestGuideFromPath:
    def test_returns_last_segment(self):
        assert guide_from_path("/services/canvas/assignments") == "assignments"

    def test_trailing_slash(self):
        assert guide_from_path("/services/canvas/courses/") == "courses"

    def test_single_segment(self):
        assert guide_from_path("/get_started") == "get_started"

    def test_empty_path(self):
        assert guide_from_path("/") is None


# ---------------------------------------------------------------------------
# is_developer_url
# ---------------------------------------------------------------------------

class TestIsDeveloperUrl:
    def test_valid_developer_url(self):
        assert is_developer_url(f"https://{ALLOWED_HOSTNAME}/services/canvas") is True

    def test_external_url_rejected(self):
        assert is_developer_url("https://other.example.com/services/canvas") is False

    def test_community_url_rejected(self):
        assert is_developer_url("https://community.instructure.com/en/kb/articles/123") is False


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_strips_fragment(self):
        url = "https://developerdocs.instructure.com/services/canvas#section"
        assert "#section" not in normalize_url(url)

    def test_strips_trailing_slash(self):
        url = "https://developerdocs.instructure.com/services/canvas/"
        assert normalize_url(url) == "https://developerdocs.instructure.com/services/canvas"


# ---------------------------------------------------------------------------
# discover_from_sitemap — integration with mocked HTTP
# ---------------------------------------------------------------------------

class TestDiscoverFromSitemap:
    @respx.mock
    def test_returns_discovery_entries(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        # Should include canvas/assignments, canvas/courses, dap/overview, studio/getting-started
        # but NOT: /, /get_started, /services, /services/canvas (index-only), other.example.com
        urls = {e.source_url for e in entries}
        assert "https://developerdocs.instructure.com/services/canvas/assignments" in urls
        assert "https://developerdocs.instructure.com/services/canvas/courses" in urls
        assert "https://developerdocs.instructure.com/services/dap/overview" in urls
        assert "https://developerdocs.instructure.com/services/studio/getting-started" in urls

    @respx.mock
    def test_excludes_index_urls(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        urls = {e.source_url for e in entries}
        assert "https://developerdocs.instructure.com/" not in urls
        assert "https://developerdocs.instructure.com/get_started" not in urls
        assert "https://developerdocs.instructure.com/services" not in urls

    @respx.mock
    def test_excludes_external_hostnames(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        urls = {e.source_url for e in entries}
        assert not any("other.example.com" in u for u in urls)

    @respx.mock
    def test_product_is_developer(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        assert all(e.product == "developer" for e in entries)

    @respx.mock
    def test_status_is_pending(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        assert all(e.status == "pending" for e in entries)

    @respx.mock
    def test_role_is_none(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        assert all(e.role is None for e in entries)

    @respx.mock
    def test_category_extracted_correctly(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        by_url = {e.source_url: e for e in entries}
        assert by_url[
            "https://developerdocs.instructure.com/services/canvas/assignments"
        ].category == "canvas"
        assert by_url[
            "https://developerdocs.instructure.com/services/dap/overview"
        ].category == "dap"

    @respx.mock
    def test_resume_skips_existing_urls(self):
        """URLs already in existing_urls must not appear in returned entries."""
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        existing = {
            "https://developerdocs.instructure.com/services/canvas/assignments",
            "https://developerdocs.instructure.com/services/dap/overview",
        }
        with httpx.Client() as client:
            entries = discover_from_sitemap(
                http_client=client,
                existing_urls=existing,
            )

        urls = {e.source_url for e in entries}
        assert "https://developerdocs.instructure.com/services/canvas/assignments" not in urls
        assert "https://developerdocs.instructure.com/services/dap/overview" not in urls
        # Other URLs should still appear
        assert "https://developerdocs.instructure.com/services/canvas/courses" in urls

    @respx.mock
    def test_resume_with_all_existing_returns_empty(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        # Pre-populate with all content URLs from the fixture
        existing = {
            "https://developerdocs.instructure.com/services/canvas/assignments",
            "https://developerdocs.instructure.com/services/canvas/courses",
            "https://developerdocs.instructure.com/services/dap/overview",
            "https://developerdocs.instructure.com/services/studio/getting-started",
        }
        with httpx.Client() as client:
            entries = discover_from_sitemap(
                http_client=client,
                existing_urls=existing,
            )

        assert entries == []

    @respx.mock
    def test_source_url_never_dropped(self):
        """Every returned entry must have a non-empty source_url."""
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        assert all(e.source_url for e in entries)

    @respx.mock
    def test_canonical_url_equals_source_url(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        assert all(e.canonical_url == e.source_url for e in entries)

    @respx.mock
    def test_guide_is_last_path_segment(self):
        respx.get(SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover_from_sitemap(http_client=client)

        by_url = {e.source_url: e for e in entries}
        assert by_url[
            "https://developerdocs.instructure.com/services/canvas/assignments"
        ].guide == "assignments"
        assert by_url[
            "https://developerdocs.instructure.com/services/studio/getting-started"
        ].guide == "getting-started"
