"""Tests for src/crawler/discover_canvas.py.

Fixtures use the real community.instructure.com URL structure:
  /en/all-guides
    → /en/kb/canvas-lms-student-guide        (collection index, 3 segments)
    → /en/kb/canvas-lms-instructor-guide
      → /en/kb/articles/{id}-{slug}          (individual article, 4 segments)
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from src.crawler.discover_canvas import (
    collection_slug_from_url,
    discover,
    extract_canonical,
    extract_links,
    extract_title,
    is_article_url,
    is_canvas_collection_url,
    is_guide_url,
    normalize_url,
)
from src.crawler.models import CANVAS_COLLECTIONS, collection_metadata

BASE = "https://community.instructure.com"


# ---------------------------------------------------------------------------
# HTML fixtures matching real site structure
# ---------------------------------------------------------------------------

ALL_GUIDES_HTML = f"""
<html><head><title>All Guides | Instructure Community</title></head>
<body>
  <a href="{BASE}/en/kb/canvas-lms-student-guide">Canvas Student Guide</a>
  <a href="{BASE}/en/kb/canvas-lms-instructor-guide">Canvas Instructor Guide</a>
  <a href="{BASE}/en/kb/canvas-lms-admin-guide">Canvas Admin Guide</a>
  <a href="{BASE}/en/kb/elevate-data-hub-guide">Elevate (not canvas)</a>
  <a href="{BASE}/en/kb/impact-guide">Impact (not canvas)</a>
  <a href="/kb/articles/663097-community-guidelines">Community Guidelines</a>
  <a href="https://other.example.com/canvas/guide">External</a>
  <a href="#fragment">Fragment-only</a>
  <a href="javascript:void(0)">JS</a>
</body></html>
"""

STUDENT_COLLECTION_HTML = f"""
<html><head><title>Canvas Student Guide | Instructure Community</title></head>
<body>
  <a href="{BASE}/en/kb/articles/661193-how-do-i-accept-an-invitation">Accept Invitation</a>
  <a href="{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment">Submit Assignment</a>
  <a href="{BASE}/en/kb/articles/661207-how-do-i-view-assignments-as-a-student">View Assignments</a>
  <a href="{BASE}/en/kb/elevate-data-hub-guide">Elevate (should not follow)</a>
</body></html>
"""

INSTRUCTOR_COLLECTION_HTML = f"""
<html><head><title>Canvas Instructor Guide | Instructure Community</title></head>
<body>
  <a href="{BASE}/en/kb/articles/660622-how-do-i-accept-an-invitation-as-an-instructor">Accept Invitation</a>
  <a href="{BASE}/en/kb/articles/660625-how-do-i-view-course-analytics">View Analytics</a>
</body></html>
"""

ARTICLE_HTML = """
<html><head>
  <title>How do I submit an online assignment? | Instructure Community</title>
  <link rel="canonical" href="https://community.instructure.com/en/kb/articles/661210-how-do-i-submit-an-online-assignment"/>
  <meta property="og:title" content="How do I submit an online assignment?"/>
</head>
<body><h1>How do I submit an online assignment?</h1></body>
</html>
"""


# ---------------------------------------------------------------------------
# Unit tests – URL filter helpers
# ---------------------------------------------------------------------------

class TestIsCanvasCollectionUrl:
    def test_student_guide_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/canvas-lms-student-guide")

    def test_instructor_guide_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/canvas-lms-instructor-guide")

    def test_admin_guide_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/canvas-lms-admin-guide")

    def test_studio_guide_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/canvas-studio-guide")

    def test_mobile_guide_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/canvas_mobile-app-ios-guide")

    def test_elevate_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/elevate-data-hub-guide")

    def test_impact_accepted(self):
        assert is_canvas_collection_url(f"{BASE}/en/kb/impact-guide")

    def test_article_url_rejected(self):
        assert not is_canvas_collection_url(f"{BASE}/en/kb/articles/661210-submit-assignment")

    def test_external_domain_rejected(self):
        assert not is_canvas_collection_url("https://example.com/en/kb/canvas-lms-student-guide")

    def test_all_guides_index_rejected(self):
        assert not is_canvas_collection_url(f"{BASE}/en/all-guides")


class TestIsArticleUrl:
    def test_article_with_slug_accepted(self):
        assert is_article_url(f"{BASE}/en/kb/articles/661210-how-do-i-submit")

    def test_article_numeric_only_accepted(self):
        assert is_article_url(f"{BASE}/en/kb/articles/661210")

    def test_collection_page_rejected(self):
        assert not is_article_url(f"{BASE}/en/kb/canvas-lms-student-guide")

    def test_all_guides_rejected(self):
        assert not is_article_url(f"{BASE}/en/all-guides")

    def test_external_domain_rejected(self):
        assert not is_article_url("https://example.com/en/kb/articles/12345")


class TestIsGuideUrl:
    def test_article_url_is_guide(self):
        assert is_guide_url(f"{BASE}/en/kb/articles/661210-submit-assignment")

    def test_collection_page_is_not_guide(self):
        assert not is_guide_url(f"{BASE}/en/kb/canvas-lms-student-guide")

    def test_all_guides_is_not_guide(self):
        assert not is_guide_url(f"{BASE}/en/all-guides")


class TestCollectionSlugFromUrl:
    def test_extracts_slug(self):
        assert collection_slug_from_url(
            f"{BASE}/en/kb/canvas-lms-student-guide"
        ) == "canvas-lms-student-guide"

    def test_returns_none_for_article(self):
        assert collection_slug_from_url(
            f"{BASE}/en/kb/articles/661210-submit"
        ) is None

    def test_returns_slug_for_elevate(self):
        assert collection_slug_from_url(f"{BASE}/en/kb/elevate-data-hub-guide") == "elevate-data-hub-guide"


class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert normalize_url(f"{BASE}/en/kb/canvas#section") == f"{BASE}/en/kb/canvas"

    def test_strips_trailing_slash(self):
        assert normalize_url(f"{BASE}/en/kb/articles/12345/") == f"{BASE}/en/kb/articles/12345"

    def test_preserves_query(self):
        url = f"{BASE}/en/kb/canvas-lms-student-guide?page=2"
        assert "page=2" in normalize_url(url)


class TestExtractLinks:
    def test_returns_canvas_collection_links(self):
        links = extract_links(ALL_GUIDES_HTML, f"{BASE}/en/all-guides")
        assert f"{BASE}/en/kb/canvas-lms-student-guide" in links

    def test_filters_external_domains(self):
        links = extract_links(ALL_GUIDES_HTML, f"{BASE}/en/all-guides")
        assert not any("other.example.com" in l for l in links)

    def test_filters_javascript_hrefs(self):
        links = extract_links(ALL_GUIDES_HTML, f"{BASE}/en/all-guides")
        assert not any("javascript" in l for l in links)

    def test_deduplicates(self):
        html = f"""<a href="{BASE}/en/kb/canvas-lms-student-guide">A</a>
                   <a href="{BASE}/en/kb/canvas-lms-student-guide">B</a>"""
        links = extract_links(html, BASE)
        assert links.count(f"{BASE}/en/kb/canvas-lms-student-guide") == 1

    def test_resolves_relative_links(self):
        html = '<a href="/en/kb/articles/661210-submit">Submit</a>'
        links = extract_links(html, f"{BASE}/en/kb/canvas-lms-student-guide")
        assert f"{BASE}/en/kb/articles/661210-submit" in links


class TestExtractTitle:
    def test_og_title_preferred(self):
        assert extract_title(ARTICLE_HTML) == "How do I submit an online assignment?"

    def test_title_tag_strips_site_name(self):
        html = "<html><head><title>My Guide | Instructure Community</title></head></html>"
        assert extract_title(html) == "My Guide"

    def test_h1_fallback(self):
        html = "<html><body><h1>My Guide</h1></body></html>"
        assert extract_title(html) == "My Guide"

    def test_returns_none_on_empty(self):
        assert extract_title("<html></html>") is None


class TestExtractCanonical:
    def test_returns_canonical_url(self):
        canon = extract_canonical(ARTICLE_HTML)
        assert canon == f"{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment"

    def test_returns_none_when_missing(self):
        assert extract_canonical("<html></html>") is None


# ---------------------------------------------------------------------------
# Unit tests – collection metadata
# ---------------------------------------------------------------------------

class TestCollectionMetadata:
    def test_student_guide_role(self):
        role, sub = collection_metadata("canvas-lms-student-guide")
        assert role == "student"
        assert sub == "lms"

    def test_instructor_guide_role(self):
        role, sub = collection_metadata("canvas-lms-instructor-guide")
        assert role == "instructor"

    def test_admin_guide_role(self):
        role, sub = collection_metadata("canvas-lms-admin-guide")
        assert role == "admin"

    def test_observer_guide_role(self):
        role, sub = collection_metadata("canvas-lms-observer-guide")
        assert role == "observer"

    def test_basics_guide_no_role(self):
        role, sub = collection_metadata("canvas-lms-basics-guide")
        assert role is None
        assert sub == "lms"

    def test_studio_guide(self):
        role, sub = collection_metadata("canvas-studio-guide")
        assert sub == "studio"

    def test_elevate_slug_metadata(self):
        role, sub = collection_metadata("elevate-data-hub-guide")
        assert role is None
        assert sub == "elevate"

    def test_unknown_slug_returns_nones(self):
        role, sub = collection_metadata("completely-unknown-slug-xyz")
        assert role is None
        assert sub is None

    def test_all_collections_registered(self):
        assert len(CANVAS_COLLECTIONS) >= 10


# ---------------------------------------------------------------------------
# Integration tests – BFS with mocked HTTP
# ---------------------------------------------------------------------------

class TestDiscover:
    @respx.mock
    def test_discovers_articles_from_collection(self):
        respx.get(f"{BASE}/en/all-guides").mock(
            return_value=httpx.Response(200, text=ALL_GUIDES_HTML)
        )
        respx.get(f"{BASE}/en/kb/canvas-lms-student-guide").mock(
            return_value=httpx.Response(200, text=STUDENT_COLLECTION_HTML)
        )
        respx.get(f"{BASE}/en/kb/canvas-lms-instructor-guide").mock(
            return_value=httpx.Response(200, text=INSTRUCTOR_COLLECTION_HTML)
        )
        respx.get(f"{BASE}/en/kb/canvas-lms-admin-guide").mock(
            return_value=httpx.Response(200, text="<html><body></body></html>")
        )
        for art_id in ["661193", "661210", "661207", "660622", "660625"]:
            respx.get(f"{BASE}/en/kb/articles/{art_id}", name=art_id).mock(
                return_value=httpx.Response(200, text=ARTICLE_HTML)
            )
        # Allow any article URL pattern
        respx.get(url__regex=r".*/en/kb/articles/.*").mock(
            return_value=httpx.Response(200, text=ARTICLE_HTML)
        )

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=3,
                max_pages=50,
                delay_s=0,
                http_client=client,
            )

        assert len(entries) > 0
        for e in entries:
            assert e.status == "pending"
            assert is_article_url(e.source_url)

    @respx.mock
    def test_role_set_from_collection(self):
        respx.get(f"{BASE}/en/all-guides").mock(
            return_value=httpx.Response(200, text=ALL_GUIDES_HTML)
        )
        respx.get(f"{BASE}/en/kb/canvas-lms-student-guide").mock(
            return_value=httpx.Response(200, text=STUDENT_COLLECTION_HTML)
        )
        respx.get(f"{BASE}/en/kb/canvas-lms-instructor-guide").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        respx.get(f"{BASE}/en/kb/canvas-lms-admin-guide").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        respx.get(url__regex=r".*/en/kb/articles/.*").mock(
            return_value=httpx.Response(200, text=ARTICLE_HTML)
        )

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=3,
                max_pages=50,
                delay_s=0,
                http_client=client,
            )

        student_entries = [e for e in entries if e.category == "canvas-lms-student-guide"]
        assert len(student_entries) > 0
        for e in student_entries:
            assert e.role == "student"

    @respx.mock
    def test_includes_all_registered_collections(self):
        respx.get(f"{BASE}/en/all-guides").mock(
            return_value=httpx.Response(200, text=ALL_GUIDES_HTML)
        )
        # elevate and impact are now registered — should be followed
        respx.get(f"{BASE}/en/kb/elevate-data-hub-guide").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        respx.get(url__regex=r".*/en/kb/canvas.*").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        respx.get(url__regex=r".*/en/kb/articles/.*").mock(
            return_value=httpx.Response(200, text=ARTICLE_HTML)
        )

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=3,
                max_pages=50,
                delay_s=0,
                http_client=client,
            )

        # articles-only entries (leaf nodes) are what we care about
        assert all(e.status == "pending" for e in entries)
        # unknown slugs still produce no entries
        urls = {e.source_url for e in entries}
        assert not any("completely-unknown-xyz" in u for u in urls)

    @respx.mock
    def test_skips_existing_urls(self):
        respx.get(f"{BASE}/en/all-guides").mock(
            return_value=httpx.Response(200, text=ALL_GUIDES_HTML)
        )
        respx.get(url__regex=r".*/en/kb/canvas.*").mock(
            return_value=httpx.Response(200, text=STUDENT_COLLECTION_HTML)
        )
        respx.get(url__regex=r".*/en/kb/articles/.*").mock(
            return_value=httpx.Response(200, text=ARTICLE_HTML)
        )

        already_found = {f"{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment"}

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=3,
                delay_s=0,
                existing_urls=already_found,
                http_client=client,
            )

        urls = {e.source_url for e in entries}
        assert f"{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment" not in urls

    @respx.mock
    def test_http_error_does_not_crash(self):
        respx.get(f"{BASE}/en/all-guides").mock(
            return_value=httpx.Response(500)
        )

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=3,
                delay_s=0,
                http_client=client,
            )

        assert entries == []

    @respx.mock
    def test_respects_max_pages(self):
        # Collection with many articles
        articles_html = "\n".join(
            f'<a href="{BASE}/en/kb/articles/100{i:03d}-guide-{i}">Guide {i}</a>'
            for i in range(50)
        )
        collection_html = f"<html><body>{articles_html}</body></html>"
        all_guides = (
            f'<html><body><a href="{BASE}/en/kb/canvas-lms-student-guide">Students</a></body></html>'
        )
        respx.get(f"{BASE}/en/all-guides").mock(return_value=httpx.Response(200, text=all_guides))
        respx.get(f"{BASE}/en/kb/canvas-lms-student-guide").mock(
            return_value=httpx.Response(200, text=collection_html)
        )
        respx.get(url__regex=r".*/en/kb/articles/.*").mock(
            return_value=httpx.Response(200, text=ARTICLE_HTML)
        )

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=3,
                max_pages=5,
                delay_s=0,
                http_client=client,
            )

        assert len(entries) <= 5

    @respx.mock
    def test_dry_run_no_file_written(self, tmp_path: Path):
        """discover() itself never writes files; caller decides."""
        respx.get(f"{BASE}/en/all-guides").mock(
            return_value=httpx.Response(200, text=ALL_GUIDES_HTML)
        )
        respx.get(url__regex=r".*community\.instructure\.com.*").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )

        out_path = tmp_path / "manifest.jsonl"

        with httpx.Client() as client:
            discover(
                start_url=f"{BASE}/en/all-guides",
                product="canvas",
                max_depth=2,
                delay_s=0,
                http_client=client,
            )

        assert not out_path.exists()

    @respx.mock
    def test_article_has_correct_guide_field(self):
        all_guides = (
            f'<html><body><a href="{BASE}/en/kb/canvas-lms-student-guide">Students</a></body></html>'
        )
        collection = (
            f'<html><body>'
            f'<a href="{BASE}/en/kb/articles/661210-how-do-i-submit-an-online-assignment">Submit</a>'
            f'</body></html>'
        )
        respx.get(f"{BASE}/en/all-guides").mock(return_value=httpx.Response(200, text=all_guides))
        respx.get(f"{BASE}/en/kb/canvas-lms-student-guide").mock(
            return_value=httpx.Response(200, text=collection)
        )
        respx.get(url__regex=r".*/en/kb/articles/.*").mock(
            return_value=httpx.Response(200, text=ARTICLE_HTML)
        )

        with httpx.Client() as client:
            entries = discover(
                start_url=f"{BASE}/en/all-guides",
                max_depth=3,
                delay_s=0,
                http_client=client,
            )

        assert len(entries) == 1
        assert entries[0].guide == "661210-how-do-i-submit-an-online-assignment"
        assert entries[0].category == "canvas-lms-student-guide"
        assert entries[0].role == "student"
        assert entries[0].title == "How do I submit an online assignment?"
