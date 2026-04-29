"""Tests for src/crawler/models.py."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.crawler.models import (
    DiscoveryEntry,
    category_from_path,
    guide_from_path,
    load_manifest,
    role_from_path,
    save_manifest,
    sha256_of,
)


def _entry(**kwargs) -> DiscoveryEntry:
    defaults = dict(
        source_url="https://community.instructure.com/en/canvas/for-students/my-guide",
        discovered_at=datetime(2026, 4, 24, 0, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return DiscoveryEntry(**defaults)


class TestDiscoveryEntry:
    def test_defaults(self):
        e = _entry()
        assert e.status == "pending"
        assert e.product == "canvas"
        assert e.content_hash is None
        assert e.retry_count == 0

    def test_jsonl_round_trip(self):
        e = _entry(title="How to Submit", role="student", category="for-students")
        line = e.to_jsonl_line()
        loaded = DiscoveryEntry.from_jsonl_line(line)
        assert loaded.source_url == e.source_url
        assert loaded.title == e.title
        assert loaded.role == e.role
        assert loaded.discovered_at == e.discovered_at

    def test_jsonl_preserves_none_fields(self):
        e = _entry()
        loaded = DiscoveryEntry.from_jsonl_line(e.to_jsonl_line())
        assert loaded.canonical_url is None
        assert loaded.fetch_error is None
        assert loaded.raw_html_path is None

    def test_status_values(self):
        for status in ("pending", "fetched", "failed", "skipped"):
            e = _entry(status=status)
            assert e.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(Exception):
            _entry(status="unknown")


class TestManifestIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        entries = [
            _entry(source_url=f"https://community.instructure.com/en/canvas/for-students/guide-{i}")
            for i in range(5)
        ]
        path = tmp_path / "manifest.jsonl"
        save_manifest(entries, path)
        loaded = load_manifest(path)
        assert len(loaded) == 5
        urls = {e.source_url for e in loaded}
        assert "https://community.instructure.com/en/canvas/for-students/guide-2" in urls

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        result = load_manifest(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "manifest.jsonl"
        save_manifest([_entry()], path)
        assert path.exists()

    def test_empty_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "manifest.jsonl"
        entry = _entry()
        path.write_text("\n" + entry.to_jsonl_line() + "\n\n", encoding="utf-8")
        loaded = load_manifest(path)
        assert len(loaded) == 1


class TestHelpers:
    def test_sha256_str_and_bytes_equal(self):
        assert sha256_of("hello") == sha256_of(b"hello")

    def test_sha256_length(self):
        assert len(sha256_of("test")) == 64

    def test_sha256_deterministic(self):
        assert sha256_of("canvas") == sha256_of("canvas")

    @pytest.mark.parametrize("path,expected", [
        ("/en/canvas/for-students/guide", "student"),
        ("/en/canvas/for-instructors/guide", "instructor"),
        ("/en/canvas/for-admins/guide", "admin"),
        ("/en/canvas/for-observers/guide", "observer"),
        ("/en/canvas/for-designers/guide", "designer"),
        ("/en/canvas/for-parents/guide", "parent"),
        ("/en/canvas", None),
        ("/en/all-guides", None),
    ])
    def test_role_from_path(self, path, expected):
        assert role_from_path(path) == expected

    @pytest.mark.parametrize("path,expected", [
        ("/en/canvas/for-students/my-guide-slug", "my-guide-slug"),
        ("/en/canvas/for-admins/admin-guide", "admin-guide"),
        ("/en/canvas/for-students", None),
        ("/en/canvas", None),
    ])
    def test_guide_from_path(self, path, expected):
        assert guide_from_path(path) == expected

    @pytest.mark.parametrize("path,expected", [
        ("/en/canvas/for-students/guide", "for-students"),
        ("/en/canvas/for-instructors/guide", "for-instructors"),
        ("/en/canvas/for-admins/guide", "for-admins"),
        ("/en/canvas", None),
        ("/en/all-guides", None),
    ])
    def test_category_from_path(self, path, expected):
        assert category_from_path(path) == expected
