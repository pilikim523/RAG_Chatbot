---
name: canvas-rag-ingest
description: Build or modify Markdown cleaning, chunking, Apple Silicon local embedding, and Qdrant indexing for Canvas Guides RAG.
argument-hint: "[task]"
allowed-tools: Read Grep Glob Bash Write Edit
---

# Canvas RAG Ingest Skill

## 목표

수집된 Canvas guide HTML을 Markdown으로 정제하고, chunk를 생성한 뒤 Qdrant `canvas_guides` 컬렉션에 적재한다.

## M3 Pro 우선 정책

- CUDA 사용 금지.
- 로컬 embedding은 Apple Silicon GPU 사용을 우선한다.
- provider 우선순위:
  1. `local_mps` — SentenceTransformers + PyTorch MPS
  2. `local_mlx` — MLX 기반 embedding 구현이 존재할 때 사용
  3. `openai` — 품질/속도/안정성 목적 fallback
- Qdrant는 Docker로 실행해도 되지만 embedding process는 macOS host에서 실행한다.

## 구현 대상

```text
src/cleaner/to_markdown.py
src/indexing/chunk.py
src/indexing/embeddings.py
src/indexing/qdrant_index.py
src/indexing/models.py
tests/test_cleaner_*.py
tests/test_chunk_*.py
tests/test_embeddings_*.py
tests/test_qdrant_*.py
```

## Embedding provider 인터페이스

```python
from typing import Protocol

class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
```

## MPS smoke test 필수

```bash
uv run python - <<'PY'
import torch
print("mps_available=", torch.backends.mps.is_available())
print("mps_built=", torch.backends.mps.is_built())
PY
```

## 완료 검증

```bash
uv run python -m src.cleaner.to_markdown \
  --input-dir data/raw_html \
  --out-dir data/markdown \
  --manifest data/manifests/canvas_docs.jsonl

uv run python -m src.indexing.chunk \
  --manifest data/manifests/canvas_docs.jsonl \
  --out data/chunks/canvas_chunks.jsonl \
  --chunk-size 900 \
  --chunk-overlap 120

EMBEDDING_PROVIDER=local_mps LOCAL_EMBEDDING_DEVICE=mps \
uv run python -m src.indexing.qdrant_index \
  --chunks data/chunks/canvas_chunks.jsonl \
  --collection canvas_guides \
  --recreate false

uv run pytest -q tests/test_cleaner*.py tests/test_chunk*.py tests/test_embeddings*.py tests/test_qdrant*.py
```
