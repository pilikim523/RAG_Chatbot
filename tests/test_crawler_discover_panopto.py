"""
Tests for src/crawler/discover_panopto.py.

All HTTP calls are intercepted with respx; no network access required.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from src.crawler.discover_panopto import (
    ALLOWED_HOSTNAME,
    ARTICLE_PREFIX,
    ROOT_SITEMAP_URL,
    SEED_URLS,
    TOPIC_PREFIX,
    _discover_bfs,
    _discover_from_sitemap,
    category_from_url,
    discover,
    extract_canonical,
    extract_links,
    extract_title,
    guide_from_url,
    is_article_or_topic_url,
    is_panopto_url,
    is_sitemap_index,
    normalize_url,
    parse_sitemap,
)
from src.crawler.models import DiscoveryEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLAIN_SITEMAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://support.panopto.com/s/article/basics-of-panopto</loc></url>
  <url><loc>https://support.panopto.com/s/article/Recording-with-Panopto</loc></url>
  <url><loc>https://support.panopto.com/s/topic/canvas-integration</loc></url>
  <url><loc>https://support.panopto.com/s/</loc></url>
  <url><loc>https://other.example.com/page</loc></url>
</urlset>
"""

SITEMAP_INDEX_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://support.panopto.com/sitemap-articles.xml</loc></sitemap>
  <sitemap><loc>https://support.panopto.com/sitemap-topics.xml</loc></sitemap>
</sitemapindex>
"""

CHILD_SITEMAP_ARTICLES_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://support.panopto.com/s/article/basics-of-panopto</loc></url>
  <url><loc>https://support.panopto.com/s/article/Editing-Your-Panopto-Recording</loc></url>
</urlset>
"""

CHILD_SITEMAP_TOPICS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://support.panopto.com/s/topic/canvas-integration</loc></url>
</urlset>
"""

ARTICLE_HTML = """\
<html>
<head>
  <title>Basics of Panopto | Panopto Support</title>
  <link rel="canonical" href="https://support.panopto.com/s/article/basics-of-panopto"/>
  <meta property="og:title" content="Basics of Panopto"/>
</head>
<body>
<h1>Basics of Panopto</h1>
<a href="/s/article/Recording-with-Panopto">Recording</a>
<a href="/s/topic/canvas-integration">Canvas</a>
<a href="https://other.example.com/page">External</a>
<a href="javascript:void(0)">JS</a>
</body>
</html>
"""

INDEX_HTML_WITH_ARTICLES = """\
<html>
<head><title>Panopto Support Home</title></head>
<body>
<a href="/s/article/basics-of-panopto">Basics</a>
<a href="/s/article/Recording-with-Panopto">Recording</a>
<a href="/s/topic/canvas-integration">Canvas Topic</a>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 1. normalize_url
# ---------------------------------------------------------------------------

def test_normalize_url_strips_fragment():
    assert normalize_url("https://support.panopto.com/s/article/foo#section") == \
        "https://support.panopto.com/s/article/foo"


def test_normalize_url_strips_trailing_slash():
    assert normalize_url("https://support.panopto.com/s/article/foo/") == \
        "https://support.panopto.com/s/article/foo"


def test_normalize_url_idempotent():
    url = "https://support.panopto.com/s/article/foo"
    assert normalize_url(url) == url


# ---------------------------------------------------------------------------
# 2. is_panopto_url
# ---------------------------------------------------------------------------

def test_is_panopto_url_correct_host():
    assert is_panopto_url("https://support.panopto.com/s/article/foo") is True


def test_is_panopto_url_wrong_host():
    assert is_panopto_url("https://developers.zoom.us/docs/api") is False


def test_is_panopto_url_subdomain_rejected():
    assert is_panopto_url("https://learn.panopto.com/s/article/foo") is False


# ---------------------------------------------------------------------------
# 3. is_article_or_topic_url
# ---------------------------------------------------------------------------

def test_is_article_url():
    assert is_article_or_topic_url("https://support.panopto.com/s/article/basics-of-panopto") is True


def test_is_topic_url():
    assert is_article_or_topic_url("https://support.panopto.com/s/topic/canvas-integration") is True


def test_is_root_not_article():
    assert is_article_or_topic_url("https://support.panopto.com/s/") is False


def test_is_external_not_article():
    assert is_article_or_topic_url("https://community.instructure.com/en/all-guides") is False


# ---------------------------------------------------------------------------
# 4. guide_from_url
# ---------------------------------------------------------------------------

def test_guide_from_article_url():
    assert guide_from_url("https://support.panopto.com/s/article/basics-of-panopto") == "basics-of-panopto"


def test_guide_from_topic_url():
    assert guide_from_url("https://support.panopto.com/s/topic/canvas-integration") == "canvas-integration"


def test_guide_from_root_url():
    # /s/ → last segment is 's'
    result = guide_from_url("https://support.panopto.com/s/")
    assert result == "s"


# ---------------------------------------------------------------------------
# 5. category_from_url
# ---------------------------------------------------------------------------

def test_category_article():
    assert category_from_url("https://support.panopto.com/s/article/basics-of-panopto") == "article"


def test_category_topic():
    assert category_from_url("https://support.panopto.com/s/topic/canvas-integration") == "topic"


def test_category_root_none():
    # /s/ → only one segment, no parent
    result = category_from_url("https://support.panopto.com/s/")
    assert result is None or result == "s"


# ---------------------------------------------------------------------------
# 6. parse_sitemap
# ---------------------------------------------------------------------------

def test_parse_sitemap_extracts_locs():
    urls = parse_sitemap(PLAIN_SITEMAP_XML)
    assert "https://support.panopto.com/s/article/basics-of-panopto" in urls
    assert "https://support.panopto.com/s/topic/canvas-integration" in urls


def test_parse_sitemap_returns_all_locs():
    urls = parse_sitemap(PLAIN_SITEMAP_XML)
    assert len(urls) == 5  # includes /s/ and external URL


# ---------------------------------------------------------------------------
# 7. is_sitemap_index
# ---------------------------------------------------------------------------

def test_is_sitemap_index_true():
    assert is_sitemap_index(SITEMAP_INDEX_XML) is True


def test_is_sitemap_index_false_for_urlset():
    assert is_sitemap_index(PLAIN_SITEMAP_XML) is False


# ---------------------------------------------------------------------------
# 8. extract_links
# ---------------------------------------------------------------------------

def test_extract_links_returns_panopto_links():
    links = extract_links(ARTICLE_HTML, "https://support.panopto.com/s/article/basics-of-panopto")
    assert any("Recording-with-Panopto" in lnk for lnk in links)
    assert any("canvas-integration" in lnk for lnk in links)


def test_extract_links_excludes_external():
    links = extract_links(ARTICLE_HTML, "https://support.panopto.com/s/article/basics-of-panopto")
    assert not any("other.example.com" in lnk for lnk in links)


def test_extract_links_excludes_javascript():
    links = extract_links(ARTICLE_HTML, "https://support.panopto.com/s/article/basics-of-panopto")
    assert not any("javascript" in lnk for lnk in links)


# ---------------------------------------------------------------------------
# 9. extract_title
# ---------------------------------------------------------------------------

def test_extract_title_og_title():
    assert extract_title(ARTICLE_HTML) == "Basics of Panopto"


def test_extract_title_returns_none_on_empty():
    assert extract_title("<html></html>") is None


# ---------------------------------------------------------------------------
# 10. extract_canonical
# ---------------------------------------------------------------------------

def test_extract_canonical():
    canonical = extract_canonical(ARTICLE_HTML)
    assert canonical == "https://support.panopto.com/s/article/basics-of-panopto"


def test_extract_canonical_missing():
    assert extract_canonical("<html><head></head><body></body></html>") is None


# ---------------------------------------------------------------------------
# 11. _discover_from_sitemap — plain sitemap success
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_from_sitemap_plain():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(200, text=PLAIN_SITEMAP_XML)
    )
    client = httpx.Client()
    entries, ok = _discover_from_sitemap(
        sitemap_url=ROOT_SITEMAP_URL,
        max_pages=100,
        delay_s=0.0,
        existing_urls=set(),
        product="panopto",
        http_client=client,
    )
    client.close()
    assert ok is True
    urls = {e.source_url for e in entries}
    assert "https://support.panopto.com/s/article/basics-of-panopto" in urls
    assert "https://support.panopto.com/s/topic/canvas-integration" in urls
    # /s/ root and external URL should NOT be included
    assert not any(u == "https://support.panopto.com/s" for u in urls)
    assert not any("other.example.com" in u for u in urls)


# ---------------------------------------------------------------------------
# 12. _discover_from_sitemap — sitemap index with child sitemaps
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_from_sitemap_index():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(200, text=SITEMAP_INDEX_XML)
    )
    respx.get("https://support.panopto.com/sitemap-articles.xml").mock(
        return_value=httpx.Response(200, text=CHILD_SITEMAP_ARTICLES_XML)
    )
    respx.get("https://support.panopto.com/sitemap-topics.xml").mock(
        return_value=httpx.Response(200, text=CHILD_SITEMAP_TOPICS_XML)
    )
    client = httpx.Client()
    entries, ok = _discover_from_sitemap(
        sitemap_url=ROOT_SITEMAP_URL,
        max_pages=100,
        delay_s=0.0,
        existing_urls=set(),
        product="panopto",
        http_client=client,
    )
    client.close()
    assert ok is True
    urls = {e.source_url for e in entries}
    assert "https://support.panopto.com/s/article/basics-of-panopto" in urls
    assert "https://support.panopto.com/s/topic/canvas-integration" in urls


# ---------------------------------------------------------------------------
# 13. _discover_from_sitemap — failure returns ([], False)
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_from_sitemap_failure():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(404, text="Not Found")
    )
    client = httpx.Client()
    entries, ok = _discover_from_sitemap(
        sitemap_url=ROOT_SITEMAP_URL,
        max_pages=100,
        delay_s=0.0,
        existing_urls=set(),
        product="panopto",
        http_client=client,
    )
    client.close()
    assert ok is False
    assert entries == []


# ---------------------------------------------------------------------------
# 14. _discover_from_sitemap — resume: existing URLs are skipped
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_from_sitemap_resume():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(200, text=PLAIN_SITEMAP_XML)
    )
    existing = {"https://support.panopto.com/s/article/basics-of-panopto"}
    client = httpx.Client()
    entries, ok = _discover_from_sitemap(
        sitemap_url=ROOT_SITEMAP_URL,
        max_pages=100,
        delay_s=0.0,
        existing_urls=existing,
        product="panopto",
        http_client=client,
    )
    client.close()
    assert ok is True
    assert not any(e.source_url == "https://support.panopto.com/s/article/basics-of-panopto" for e in entries)


# ---------------------------------------------------------------------------
# 15. _discover_bfs — basic BFS from seed
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_bfs_collects_articles():
    # normalize_url strips the trailing slash, so /s/ becomes /s
    respx.get("https://support.panopto.com/s").mock(
        return_value=httpx.Response(200, text=INDEX_HTML_WITH_ARTICLES)
    )
    respx.get("https://support.panopto.com/s/article/basics-of-panopto").mock(
        return_value=httpx.Response(200, text=ARTICLE_HTML)
    )
    respx.get("https://support.panopto.com/s/article/Recording-with-Panopto").mock(
        return_value=httpx.Response(200, text=ARTICLE_HTML)
    )
    respx.get("https://support.panopto.com/s/topic/canvas-integration").mock(
        return_value=httpx.Response(200, text=ARTICLE_HTML)
    )
    client = httpx.Client()
    entries = _discover_bfs(
        seed_urls=["https://support.panopto.com/s/"],
        max_pages=50,
        delay_s=0.0,
        existing_urls=set(),
        product="panopto",
        http_client=client,
        max_depth=2,
    )
    client.close()
    urls = {e.source_url for e in entries}
    assert "https://support.panopto.com/s/article/basics-of-panopto" in urls
    assert "https://support.panopto.com/s/article/Recording-with-Panopto" in urls


# ---------------------------------------------------------------------------
# 16. _discover_bfs — fetch failure records URL as pending
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_bfs_fetch_failure_records_pending():
    respx.get("https://support.panopto.com/s/article/basics-of-panopto").mock(
        return_value=httpx.Response(500, text="Server Error")
    )
    client = httpx.Client()
    entries = _discover_bfs(
        seed_urls=["https://support.panopto.com/s/article/basics-of-panopto"],
        max_pages=10,
        delay_s=0.0,
        existing_urls=set(),
        product="panopto",
        http_client=client,
    )
    client.close()
    # The URL must still appear as pending, not silently dropped
    assert any(e.source_url == "https://support.panopto.com/s/article/basics-of-panopto" for e in entries)
    assert all(e.status == "pending" for e in entries)


# ---------------------------------------------------------------------------
# 17. discover — sitemap success path used
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_uses_sitemap_when_available():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(200, text=PLAIN_SITEMAP_XML)
    )
    client = httpx.Client()
    entries = discover(
        sitemap_url=ROOT_SITEMAP_URL,
        product="panopto",
        max_pages=100,
        delay_s=0.0,
        existing_urls=set(),
        http_client=client,
    )
    client.close()
    assert len(entries) >= 2
    assert all(e.product == "panopto" for e in entries)
    assert all(e.status == "pending" for e in entries)


# ---------------------------------------------------------------------------
# 18. discover — BFS fallback when sitemap fails
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_bfs_fallback_on_sitemap_failure():
    respx.get(ROOT_SITEMAP_URL).mock(return_value=httpx.Response(404, text=""))
    respx.get("https://support.panopto.com/s/article/basics-of-panopto").mock(
        return_value=httpx.Response(200, text=ARTICLE_HTML)
    )
    # Mock all other seed URLs as 404 to keep test deterministic
    for su in SEED_URLS:
        if su != "https://support.panopto.com/s/article/basics-of-panopto":
            respx.get(su).mock(return_value=httpx.Response(404, text=""))

    client = httpx.Client()
    entries = discover(
        sitemap_url=ROOT_SITEMAP_URL,
        seed_urls=["https://support.panopto.com/s/article/basics-of-panopto"],
        product="panopto",
        max_pages=10,
        delay_s=0.0,
        existing_urls=set(),
        http_client=client,
    )
    client.close()
    assert any(e.source_url == "https://support.panopto.com/s/article/basics-of-panopto" for e in entries)


# ---------------------------------------------------------------------------
# 19. discover — max_pages is honoured
# ---------------------------------------------------------------------------

@respx.mock
def test_discover_respects_max_pages():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(200, text=PLAIN_SITEMAP_XML)
    )
    client = httpx.Client()
    entries = discover(
        sitemap_url=ROOT_SITEMAP_URL,
        product="panopto",
        max_pages=1,
        delay_s=0.0,
        existing_urls=set(),
        http_client=client,
    )
    client.close()
    assert len(entries) <= 1


# ---------------------------------------------------------------------------
# 20. DiscoveryEntry fields are correctly set
# ---------------------------------------------------------------------------

@respx.mock
def test_discovery_entry_fields():
    respx.get(ROOT_SITEMAP_URL).mock(
        return_value=httpx.Response(200, text=PLAIN_SITEMAP_XML)
    )
    client = httpx.Client()
    entries = discover(
        sitemap_url=ROOT_SITEMAP_URL,
        product="panopto",
        max_pages=100,
        delay_s=0.0,
        existing_urls=set(),
        http_client=client,
    )
    client.close()

    article_entry = next(
        e for e in entries
        if e.source_url == "https://support.panopto.com/s/article/basics-of-panopto"
    )
    assert article_entry.guide == "basics-of-panopto"
    assert article_entry.category == "article"
    assert article_entry.role is None
    assert article_entry.product == "panopto"
    assert article_entry.source_url is not None  # never silently dropped

    topic_entry = next(
        e for e in entries
        if e.source_url == "https://support.panopto.com/s/topic/canvas-integration"
    )
    assert topic_entry.guide == "canvas-integration"
    assert topic_entry.category == "topic"
