"""
Stage 4a: Text embedding with GPU-first strategy.

Priority:
  1. SentenceTransformers on CUDA (Linux/Windows with NVIDIA GPU)
  2. SentenceTransformers on MPS (Apple Silicon)
  3. SentenceTransformers on CPU
  4. OpenAI text-embedding-3-small (fallback / when ST not installed)
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


DEFAULT_ST_MODEL = "BAAI/bge-m3"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_DIM = 1536


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# MPS / CPU SentenceTransformers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


class MpsEmbedder(BaseEmbedder):
    """SentenceTransformers embedder, MPS-first with CPU fallback."""

    def __init__(
        self,
        model_name: str = DEFAULT_ST_MODEL,
        encode_batch_size: int = 1,    # 1로 설정 — 긴 문서 OOM 방지 (bge-m3 MPS)
        _model: Any = None,            # injection point for tests
    ) -> None:
        self._model_name = model_name
        self._encode_batch_size = encode_batch_size
        self.__model = _model      # pre-injected model (avoids lazy load)

    @property
    def _st_model(self) -> Any:
        if self.__model is None:
            from sentence_transformers import SentenceTransformer
            if _cuda_available():
                device = "cuda"
            elif _mps_available():
                device = "mps"
            else:
                device = "cpu"
            import logging
            logging.getLogger(__name__).info("SentenceTransformer device: %s", device)
            self.__model = SentenceTransformer(self._model_name, device=device)
        return self.__model

    @property
    def dim(self) -> int:
        return int(self._st_model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._st_model.encode(
            texts,
            batch_size=self._encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# OpenAI fallback
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small embedder."""

    def __init__(
        self,
        model: str = OPENAI_EMBED_MODEL,
        api_key: str | None = None,
        _client: Any = None,       # injection point for tests
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.__client = _client

    @property
    def _openai_client(self) -> Any:
        if self.__client is None:
            from openai import OpenAI
            self.__client = OpenAI(api_key=self._api_key)
        return self.__client

    @property
    def dim(self) -> int:
        return OPENAI_EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._openai_client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(
    prefer: str = "mps",
    openai_api_key: str | None = None,
) -> BaseEmbedder:
    """Return the best available embedder.

    prefer="mps"    → try SentenceTransformers (MPS or CPU), raise if unavailable
    prefer="openai" → always return OpenAIEmbedder
    prefer="auto"   → SentenceTransformers if installed, else OpenAI
    """
    if prefer == "openai":
        return OpenAIEmbedder(api_key=openai_api_key)

    try:
        import sentence_transformers  # noqa: F401
        return MpsEmbedder()
    except ImportError:
        if prefer == "mps":
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: uv sync --group embedding\n"
                "Or pass prefer='openai' to use OpenAI embeddings."
            )
        return OpenAIEmbedder(api_key=openai_api_key)
