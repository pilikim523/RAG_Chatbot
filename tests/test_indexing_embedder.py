"""Tests for src/indexing/embedder.py — no real GPU required."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.indexing.embedder import (
    MpsEmbedder,
    OpenAIEmbedder,
    _mps_available,
    get_embedder,
)


# ---------------------------------------------------------------------------
# Fake ST model for injection
# ---------------------------------------------------------------------------

class _FakeSTModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True, batch_size=32):
        import numpy as np
        return np.array([[float(i) for i in range(4)] for _ in texts])


# ---------------------------------------------------------------------------
# MpsEmbedder
# ---------------------------------------------------------------------------

class TestMpsEmbedder:
    def _embedder(self) -> MpsEmbedder:
        return MpsEmbedder(_model=_FakeSTModel())

    def test_dim(self):
        assert self._embedder().dim == 4

    def test_embed_returns_list_of_lists(self):
        vecs = self._embedder().embed(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 4

    def test_embed_empty_returns_empty(self):
        assert self._embedder().embed([]) == []

    def test_embed_one_returns_single_vector(self):
        vec = self._embedder().embed_one("hello")
        assert len(vec) == 4

    def test_values_are_floats(self):
        vecs = self._embedder().embed(["test"])
        assert all(isinstance(v, float) for v in vecs[0])

    def test_lazy_load_not_triggered_with_injection(self):
        """Injected model means no import of sentence_transformers."""
        emb = MpsEmbedder(_model=_FakeSTModel())
        # Access _st_model — should return injected model
        assert emb._st_model is not None


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------

def _make_openai_response(n: int, dim: int = 8):
    items = []
    for i in range(n):
        item = MagicMock()
        item.embedding = [float(j) for j in range(dim)]
        items.append(item)
    resp = MagicMock()
    resp.data = items
    return resp


class TestOpenAIEmbedder:
    def _embedder(self, dim: int = 8) -> OpenAIEmbedder:
        client = MagicMock()
        client.embeddings.create.return_value = _make_openai_response(1, dim)
        return OpenAIEmbedder(_client=client)

    def test_dim_is_1536(self):
        emb = OpenAIEmbedder(_client=MagicMock())
        assert emb.dim == 1536

    def test_embed_calls_api(self):
        client = MagicMock()
        client.embeddings.create.return_value = _make_openai_response(2, 8)
        emb = OpenAIEmbedder(_client=client)
        vecs = emb.embed(["a", "b"])
        assert len(vecs) == 2
        client.embeddings.create.assert_called_once()

    def test_embed_empty_returns_empty(self):
        emb = OpenAIEmbedder(_client=MagicMock())
        assert emb.embed([]) == []

    def test_embed_one(self):
        client = MagicMock()
        client.embeddings.create.return_value = _make_openai_response(1, 8)
        emb = OpenAIEmbedder(_client=client)
        vec = emb.embed_one("hello")
        assert len(vec) == 8


# ---------------------------------------------------------------------------
# _mps_available
# ---------------------------------------------------------------------------

class TestMpsAvailable:
    def test_returns_bool(self):
        result = _mps_available()
        assert isinstance(result, bool)

    def test_false_when_torch_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert _mps_available() is False


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------

class TestGetEmbedder:
    def test_prefer_openai_returns_openai(self):
        emb = get_embedder(prefer="openai", openai_api_key="test-key")
        assert isinstance(emb, OpenAIEmbedder)

    def test_prefer_auto_falls_back_to_openai_when_st_missing(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            emb = get_embedder(prefer="auto", openai_api_key="test-key")
            assert isinstance(emb, OpenAIEmbedder)

    def test_prefer_mps_raises_when_st_missing(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
                get_embedder(prefer="mps")

    def test_prefer_mps_returns_mps_when_st_installed(self):
        fake_st_module = MagicMock()
        fake_st_module.SentenceTransformer = MagicMock(return_value=_FakeSTModel())
        with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
            emb = get_embedder(prefer="mps")
            assert isinstance(emb, MpsEmbedder)
