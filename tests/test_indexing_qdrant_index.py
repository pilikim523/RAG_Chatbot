"""Tests for src/indexing/qdrant_index.py — no real Qdrant required."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.indexing.embedder import BaseEmbedder
from src.indexing.models import Chunk, make_chunk_id
from src.indexing.qdrant_index import (
    _chunk_payload,
    chunk_to_point_id,
    ensure_collection,
    get_existing_hashes,
    upsert_chunks,
)
from src.crawler.models import sha256_of

BASE_URL = "https://community.instructure.com/en/kb/articles/661210-submit"


# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------

class FakeEmbedder(BaseEmbedder):
    """Returns deterministic 4-dim vectors for test isolation."""

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 4) for i in range(4)] for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(idx: int = 0, total: int = 3, text: str = "Sample text.") -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(BASE_URL, idx),
        source_url=BASE_URL,
        title="Submit Assignment",
        product="canvas",
        role="student",
        category="canvas-lms-student-guide",
        chunk_index=idx,
        chunk_total=total,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
        content_hash=sha256_of(text),
    )


def _make_qdrant_client(existing_hashes: list[str] | None = None) -> MagicMock:
    """Return a MagicMock QdrantClient."""
    client = MagicMock()

    # get_collections
    col_mock = MagicMock()
    col_mock.name = "other_collection"
    client.get_collections.return_value.collections = [col_mock]

    # scroll — returns (points, None) indicating end of scroll
    if existing_hashes:
        points = []
        for h in existing_hashes:
            p = MagicMock()
            p.payload = {"content_hash": h}
            points.append(p)
        client.scroll.return_value = (points, None)
    else:
        client.scroll.return_value = ([], None)

    return client


# ---------------------------------------------------------------------------
# chunk_to_point_id
# ---------------------------------------------------------------------------

class TestChunkToPointId:
    def test_returns_int(self):
        cid = make_chunk_id(BASE_URL, 0)
        result = chunk_to_point_id(cid)
        assert isinstance(result, int)

    def test_deterministic(self):
        cid = make_chunk_id(BASE_URL, 0)
        assert chunk_to_point_id(cid) == chunk_to_point_id(cid)

    def test_different_ids_different_ints(self):
        id0 = chunk_to_point_id(make_chunk_id(BASE_URL, 0))
        id1 = chunk_to_point_id(make_chunk_id(BASE_URL, 1))
        assert id0 != id1

    def test_valid_uint64(self):
        cid = make_chunk_id(BASE_URL, 0)
        val = chunk_to_point_id(cid)
        assert 0 <= val < 2**64


# ---------------------------------------------------------------------------
# _chunk_payload
# ---------------------------------------------------------------------------

class TestChunkPayload:
    def test_has_required_fields(self):
        chunk = _make_chunk()
        payload = _chunk_payload(chunk)
        for field in ("source_url", "title", "role", "category", "product",
                      "chunk_index", "chunk_total", "text", "content_hash"):
            assert field in payload

    def test_text_preserved(self):
        chunk = _make_chunk(text="Hello world!")
        assert _chunk_payload(chunk)["text"] == "Hello world!"

    def test_content_hash_preserved(self):
        chunk = _make_chunk()
        assert _chunk_payload(chunk)["content_hash"] == chunk.content_hash


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------

class TestEnsureCollection:
    def test_creates_collection_if_missing(self):
        client = _make_qdrant_client()
        ensure_collection(client, "canvas_guides", dim=4)
        client.create_collection.assert_called_once()

    def test_does_not_create_if_exists(self):
        client = MagicMock()
        col = MagicMock()
        col.name = "canvas_guides"
        client.get_collections.return_value.collections = [col]
        ensure_collection(client, "canvas_guides", dim=4)
        client.create_collection.assert_not_called()

    def test_creates_payload_indexes(self):
        client = _make_qdrant_client()
        ensure_collection(client, "canvas_guides", dim=4)
        assert client.create_payload_index.call_count >= 6  # 6 fields


# ---------------------------------------------------------------------------
# get_existing_hashes
# ---------------------------------------------------------------------------

class TestGetExistingHashes:
    def test_returns_set_of_hashes(self):
        hashes = ["abc123", "def456"]
        client = _make_qdrant_client(existing_hashes=hashes)
        result = get_existing_hashes(client, "canvas_guides")
        assert result == set(hashes)

    def test_empty_collection_returns_empty_set(self):
        client = _make_qdrant_client()
        result = get_existing_hashes(client, "canvas_guides")
        assert result == set()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------

class TestUpsertChunks:
    def test_upserts_all_new_chunks(self):
        chunks = [_make_chunk(i, total=3) for i in range(3)]
        client = _make_qdrant_client()
        embedder = FakeEmbedder()
        stats = upsert_chunks(chunks, embedder, client, collection="canvas_guides")
        assert stats["upserted"] == 3
        assert stats["skipped"] == 0
        assert stats["total"] == 3

    def test_skips_existing_hashes(self):
        chunks = [_make_chunk(i, total=3, text=f"Unique text for chunk {i}.") for i in range(3)]
        existing = [chunks[0].content_hash]
        client = _make_qdrant_client(existing_hashes=existing)
        embedder = FakeEmbedder()
        stats = upsert_chunks(chunks, embedder, client, collection="canvas_guides")
        assert stats["upserted"] == 2
        assert stats["skipped"] == 1

    def test_skip_existing_false_upserts_all(self):
        chunks = [_make_chunk(i, total=3) for i in range(3)]
        client = _make_qdrant_client()
        embedder = FakeEmbedder()
        stats = upsert_chunks(
            chunks, embedder, client,
            collection="canvas_guides",
            skip_existing=False,
        )
        assert stats["upserted"] == 3

    def test_empty_chunks(self):
        client = _make_qdrant_client()
        embedder = FakeEmbedder()
        stats = upsert_chunks([], embedder, client, collection="canvas_guides")
        assert stats["upserted"] == 0
        assert stats["total"] == 0

    def test_calls_upsert_with_correct_batch(self):
        chunks = [_make_chunk(i, total=5) for i in range(5)]
        client = _make_qdrant_client()
        embedder = FakeEmbedder()
        upsert_chunks(chunks, embedder, client, collection="canvas_guides", batch_size=3)
        # 5 chunks / batch_size=3 → 2 upsert calls
        assert client.upsert.call_count == 2

    def test_point_ids_are_uint64(self):
        chunks = [_make_chunk(0, total=1)]
        client = _make_qdrant_client()
        embedder = FakeEmbedder()
        upsert_chunks(chunks, embedder, client, collection="canvas_guides")
        upsert_call = client.upsert.call_args
        points = upsert_call[1]["points"] if upsert_call[1] else upsert_call[0][1]
        for p in points:
            assert isinstance(p.id, int)
            assert 0 <= p.id < 2**64

    def test_vector_length_matches_embedder_dim(self):
        chunks = [_make_chunk(0, total=1)]
        client = _make_qdrant_client()
        embedder = FakeEmbedder()
        upsert_chunks(chunks, embedder, client, collection="canvas_guides")
        upsert_call = client.upsert.call_args
        points = upsert_call[1]["points"] if upsert_call[1] else upsert_call[0][1]
        assert len(points[0].vector) == embedder.dim
