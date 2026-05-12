"""
Tests for src/crawler/discover_zoom.py.

All HTTP calls are intercepted with respx; no network access required.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from src.crawler.discover_zoom import (
    ALLOWED_HOSTNAME,
    ROOT_SITEMAP_URL,
    category_from_url,
    discover,
    guide_from_url,
    has_url_prefix,
    is_sitemap_index,
    is_zoom_url,
    normalize_url,
    parse_sitemap,
)
from src.crawler.models import DiscoveryEntry

# ---------------------------------------------------------------------------
# Sample sitemap XML fixtures
# ---------------------------------------------------------------------------

SITEMAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://developers.zoom.us/</loc></url>
  <url><loc>https://developers.zoom.us/docs/api/</loc></url>
  <url><loc>https://developers.zoom.us/docs/api/meetings/create</loc></url>
  <url><loc>https://developers.zoom.us/docs/api/webinars/list</loc></url>
  <url><loc>https://developers.zoom.us/docs/zoom-apps/introduction</loc></url>
  <url><loc>https://developers.zoom.us/changelog</loc></url>
  <url><loc>https://other.example.com/docs/something</loc></url>
</urlset>
"""

SITEMAP_INDEX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://developers.zoom.us/sitemap-docs.xml</loc></sitemap>
  <sitemap><loc>https://developers.zoom.us/sitemap-changelog.xml</loc></sitemap>
</sitemapindex>
"""

CHILD_SITEMAP_DOCS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://developers.zoom.us/docs/api/meetings/create</loc></url>
  <url><loc>https://developers.zoom.us/docs/zoom-apps/introduction</loc></url>
</urlset>
"""

CHILD_SITEMAP_CHANGELOG_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://developers.zoom.us/changelog/2024-01</loc></url>
</urlset>
"""

SITEMAP_NO_NS = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url><loc>https://developers.zoom.us/docs/api/meetings/create</loc></url>
  <url><loc>https://developers.zoom.us/docs/zoom-apps/introduction</loc></url>
</urlset>
"""


# ---------------------------------------------------------------------------
# parse_sitemap
# ---------------------------------------------------------------------------

class TestParseSitemap:
    def test_returns_all_loc_urls(self):
        urls = parse_sitemap(SITEMAP_XML)
        assert "https://developers.zoom.us/docs/api/meetings/create" in urls
        assert "https://other.example.com/docs/something" in urls

    def test_count(self):
        urls = parse_sitemap(SITEMAP_XML)
        assert len(urls) == 7

    def test_no_namespace(self):
        urls = parse_sitemap(SITEMAP_NO_NS)
        assert len(urls) == 2
        assert "https://developers.zoom.us/docs/api/meetings/create" in urls

    def test_sitemap_index_locs(self):
        urls = parse_sitemap(SITEMAP_INDEX_XML)
        assert "https://developers.zoom.us/sitemap-docs.xml" in urls
        assert "https://developers.zoom.us/sitemap-changelog.xml" in urls


# ---------------------------------------------------------------------------
# is_sitemap_index
# ---------------------------------------------------------------------------

class TestIsSitemapIndex:
    def test_index_detected(self):
        assert is_sitemap_index(SITEMAP_INDEX_XML) is True

    def test_plain_sitemap_not_index(self):
        assert is_sitemap_index(SITEMAP_XML) is False

    def test_no_namespace_sitemap_not_index(self):
        assert is_sitemap_index(SITEMAP_NO_NS) is False


# ---------------------------------------------------------------------------
# is_zoom_url
# ---------------------------------------------------------------------------

class TestIsZoomUrl:
    def test_valid_zoom_url(self):
        assert is_zoom_url(f"https://{ALLOWED_HOSTNAME}/docs/api") is True

    def test_external_url_rejected(self):
        assert is_zoom_url("https://other.example.com/docs/api") is False

    def test_community_url_rejected(self):
        assert is_zoom_url("https://community.instructure.com/en/kb/articles/123") is False


# ---------------------------------------------------------------------------
# has_url_prefix
# ---------------------------------------------------------------------------

class TestHasUrlPrefix:
    def test_matching_prefix(self):
        assert has_url_prefix("https://developers.zoom.us/docs/api", "/docs/") is True

    def test_non_matching_prefix(self):
        assert has_url_prefix("https://developers.zoom.us/changelog", "/docs/") is False

    def test_root_url_does_not_match(self):
        assert has_url_prefix("https://developers.zoom.us/", "/docs/") is False


# ---------------------------------------------------------------------------
# guide_from_url
# ---------------------------------------------------------------------------

class TestGuideFromUrl:
    def test_returns_last_segment(self):
        assert guide_from_url("https://developers.zoom.us/docs/api/meetings/create") == "create"

    def test_trailing_slash(self):
        assert guide_from_url("https://developers.zoom.us/docs/api/") == "api"

    def test_root_returns_none(self):
        assert guide_from_url("https://developers.zoom.us/") is None


# ---------------------------------------------------------------------------
# category_from_url
# ---------------------------------------------------------------------------

class TestCategoryFromUrl:
    def test_extracts_first_segment_under_prefix(self):
        assert category_from_url(
            "https://developers.zoom.us/docs/api/meetings/create", "/docs/"
        ) == "api"

    def test_zoom_apps_category(self):
        assert category_from_url(
            "https://developers.zoom.us/docs/zoom-apps/introduction", "/docs/"
        ) == "zoom-apps"

    def test_single_segment_under_prefix(self):
        assert category_from_url(
            "https://developers.zoom.us/docs/api", "/docs/"
        ) == "api"


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_strips_fragment(self):
        url = "https://developers.zoom.us/docs/api#section"
        assert "#section" not in normalize_url(url)

    def test_strips_trailing_slash(self):
        assert normalize_url(
            "https://developers.zoom.us/docs/api/"
        ) == "https://developers.zoom.us/docs/api"


# ---------------------------------------------------------------------------
# discover — integration with mocked HTTP (plain sitemap)
# ---------------------------------------------------------------------------

class TestDiscover:
    @respx.mock
    def test_returns_docs_entries(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        urls = {e.source_url for e in entries}
        assert "https://developers.zoom.us/docs/api/meetings/create" in urls
        assert "https://developers.zoom.us/docs/api/webinars/list" in urls
        assert "https://developers.zoom.us/docs/zoom-apps/introduction" in urls

    @respx.mock
    def test_excludes_non_docs_paths(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        urls = {e.source_url for e in entries}
        # /changelog and / are outside /docs/
        assert "https://developers.zoom.us/changelog" not in urls
        assert "https://developers.zoom.us/" not in urls

    @respx.mock
    def test_excludes_external_hostnames(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        urls = {e.source_url for e in entries}
        assert not any("other.example.com" in u for u in urls)

    @respx.mock
    def test_product_is_zoom(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        assert all(e.product == "zoom" for e in entries)

    @respx.mock
    def test_status_is_pending(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        assert all(e.status == "pending" for e in entries)

    @respx.mock
    def test_role_is_none(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        assert all(e.role is None for e in entries)

    @respx.mock
    def test_source_url_never_dropped(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        assert all(e.source_url for e in entries)

    @respx.mock
    def test_canonical_url_equals_source_url(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        assert all(e.canonical_url == e.source_url for e in entries)

    @respx.mock
    def test_guide_is_last_path_segment(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        by_url = {e.source_url: e for e in entries}
        assert by_url["https://developers.zoom.us/docs/api/meetings/create"].guide == "create"
        assert by_url["https://developers.zoom.us/docs/zoom-apps/introduction"].guide == "introduction"

    @respx.mock
    def test_category_extracted_correctly(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        by_url = {e.source_url: e for e in entries}
        assert by_url["https://developers.zoom.us/docs/api/meetings/create"].category == "api"
        assert by_url["https://developers.zoom.us/docs/zoom-apps/introduction"].category == "zoom-apps"

    @respx.mock
    def test_resume_skips_existing_urls(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        existing = {
            "https://developers.zoom.us/docs/api/meetings/create",
            "https://developers.zoom.us/docs/api/webinars/list",
        }
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0, existing_urls=existing)

        urls = {e.source_url for e in entries}
        assert "https://developers.zoom.us/docs/api/meetings/create" not in urls
        assert "https://developers.zoom.us/docs/api/webinars/list" not in urls
        assert "https://developers.zoom.us/docs/zoom-apps/introduction" in urls

    @respx.mock
    def test_resume_all_existing_returns_empty(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        # /docs/api/ normalizes to /docs/api (trailing slash stripped)
        existing = {
            "https://developers.zoom.us/docs/api",
            "https://developers.zoom.us/docs/api/meetings/create",
            "https://developers.zoom.us/docs/api/webinars/list",
            "https://developers.zoom.us/docs/zoom-apps/introduction",
        }
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0, existing_urls=existing)

        assert entries == []

    @respx.mock
    def test_max_pages_respected(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0, max_pages=2)

        assert len(entries) <= 2

    @respx.mock
    def test_custom_url_prefix(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_XML)
        )
        with httpx.Client() as client:
            entries = discover(
                http_client=client,
                delay_s=0,
                url_prefix="/docs/api",
            )

        urls = {e.source_url for e in entries}
        # Only /docs/api/* should be included; /docs/zoom-apps is excluded
        assert all("/docs/api" in u for u in urls)
        assert "https://developers.zoom.us/docs/zoom-apps/introduction" not in urls


# ---------------------------------------------------------------------------
# discover — sitemap index traversal
# ---------------------------------------------------------------------------

class TestDiscoverSitemapIndex:
    @respx.mock
    def test_traverses_child_sitemaps(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_INDEX_XML)
        )
        respx.get("https://developers.zoom.us/sitemap-docs.xml").mock(
            return_value=httpx.Response(200, text=CHILD_SITEMAP_DOCS_XML)
        )
        respx.get("https://developers.zoom.us/sitemap-changelog.xml").mock(
            return_value=httpx.Response(200, text=CHILD_SITEMAP_CHANGELOG_XML)
        )

        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        urls = {e.source_url for e in entries}
        # From docs child sitemap
        assert "https://developers.zoom.us/docs/api/meetings/create" in urls
        assert "https://developers.zoom.us/docs/zoom-apps/introduction" in urls
        # changelog is outside /docs/ prefix, excluded
        assert "https://developers.zoom.us/changelog/2024-01" not in urls

    @respx.mock
    def test_child_sitemap_fetch_failure_is_non_fatal(self):
        respx.get(ROOT_SITEMAP_URL).mock(
            return_value=httpx.Response(200, text=SITEMAP_INDEX_XML)
        )
        respx.get("https://developers.zoom.us/sitemap-docs.xml").mock(
            return_value=httpx.Response(200, text=CHILD_SITEMAP_DOCS_XML)
        )
        # Second child fails
        respx.get("https://developers.zoom.us/sitemap-changelog.xml").mock(
            return_value=httpx.Response(500)
        )

        with httpx.Client() as client:
            entries = discover(http_client=client, delay_s=0)

        # Should still return entries from the successful child
        assert len(entries) > 0
        urls = {e.source_url for e in entries}
        assert "https://developers.zoom.us/docs/api/meetings/create" in urls
