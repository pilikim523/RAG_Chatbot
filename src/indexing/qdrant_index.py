"""
Stage 4b: Qdrant ingestion.

Reads canvas_chunks.jsonl, embeds with the configured embedder,
and upserts into a Qdrant collection. Idempotent: skips chunks
whose content_hash already exists in the collection.

Usage:
    uv run python -m src.indexing.qdrant_index \\
        --chunks data/chunks/canvas_chunks.jsonl \\
        --collection canvas_guides \\
        --qdrant-url http://localhost:6333
"""
from __future__ import annotations

import time
from typing import Any

import click
from rich.console import Console
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.indexing.embedder import BaseEmbedder, get_embedder
from src.indexing.models import Chunk, load_chunks

console = Console(stderr=True)

COLLECTION_NAME = "canvas_guides"
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def chunk_to_point_id(chunk_id: str) -> int:
    """Convert 16-hex chunk_id to uint64 Qdrant point ID."""
    return int(chunk_id, 16)


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "source_url": chunk.source_url,
        "canonical_url": chunk.canonical_url,
        "title": chunk.title,
        "product": chunk.product,
        "guide": chunk.guide,
        "category": chunk.category,
        "role": chunk.role,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "text": chunk.text,
        "char_count": chunk.char_count,
        "word_count": chunk.word_count,
        "content_hash": chunk.content_hash,
    }


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """Create collection + payload indexes if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        console.print(f"[green]Created collection '{collection}' (dim={dim})[/green]")

    # Keyword indexes for filtered search
    for field in ("source_url", "guide", "category", "role", "product", "content_hash"):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index already exists


def get_existing_hashes(client: QdrantClient, collection: str) -> set[str]:
    """Scroll the collection and return all content_hash values."""
    hashes: set[str] = set()
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            with_payload=["content_hash"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in result:
            h = (point.payload or {}).get("content_hash")
            if h:
                hashes.add(h)
        if offset is None:
            break
    return hashes


def upsert_chunks(
    chunks: list[Chunk],
    embedder: BaseEmbedder,
    client: QdrantClient,
    collection: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Embed and upsert chunks. Returns stats dict."""
    ensure_collection(client, collection, embedder.dim)

    existing_hashes: set[str] = set()
    if skip_existing:
        existing_hashes = get_existing_hashes(client, collection)
        console.print(f"[dim]{len(existing_hashes)} existing hashes in collection[/dim]")

    new_chunks = [c for c in chunks if c.content_hash not in existing_hashes]
    skipped = len(chunks) - len(new_chunks)
    console.print(f"[blue]{len(new_chunks)} new chunks to upsert, {skipped} skipped (already indexed)[/blue]")

    upserted = 0
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        vectors = embedder.embed(texts)

        points = [
            PointStruct(
                id=chunk_to_point_id(c.chunk_id),
                vector=v,
                payload=_chunk_payload(c),
            )
            for c, v in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        upserted += len(points)
        console.print(f"[dim]  upserted batch {i // batch_size + 1}: {upserted}/{len(new_chunks)}[/dim]")

    return {"total": len(chunks), "upserted": upserted, "skipped": skipped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--chunks", required=True, type=click.Path(exists=True),
    help="Chunks JSONL file (canvas_chunks.jsonl)",
)
@click.option(
    "--collection", default=COLLECTION_NAME, show_default=True,
    help="Qdrant collection name",
)
@click.option(
    "--qdrant-url", default="http://localhost:6333", show_default=True,
    help="Qdrant server URL",
)
@click.option(
    "--embedder", "embedder_type", default="auto",
    type=click.Choice(["mps", "openai", "auto"]), show_default=True,
    help="Embedder to use",
)
@click.option(
    "--batch-size", default=BATCH_SIZE, show_default=True,
    help="Chunks per embedding batch",
)
@click.option(
    "--no-skip", is_flag=True, default=False,
    help="Re-embed and overwrite all chunks (ignore existing hashes)",
)
@click.option("--dry-run", is_flag=True, help="Embed but do not upsert to Qdrant")
def main(
    chunks: str,
    collection: str,
    qdrant_url: str,
    embedder_type: str,
    batch_size: int,
    no_skip: bool,
    dry_run: bool,
) -> None:
    """Embed Canvas guide chunks and upsert into Qdrant."""
    all_chunks = load_chunks_file(chunks)
    console.print(f"[blue]Loaded {len(all_chunks)} chunks from {chunks}[/blue]")

    embedder = get_embedder(prefer=embedder_type)
    console.print(f"[blue]Embedder: {type(embedder).__name__}, dim={embedder.dim}[/blue]")

    if dry_run:
        console.print("[yellow]--dry-run: embedding first batch only, no upsert[/yellow]")
        sample = all_chunks[:min(batch_size, len(all_chunks))]
        vecs = embedder.embed([c.text for c in sample])
        console.print(f"[green]Embedded {len(vecs)} sample vectors (dim={len(vecs[0]) if vecs else 0})[/green]")
        return

    client = QdrantClient(url=qdrant_url)
    stats = upsert_chunks(
        all_chunks, embedder, client,
        collection=collection,
        batch_size=batch_size,
        skip_existing=not no_skip,
    )
    console.print(
        f"[bold green]Done. upserted={stats['upserted']}, "
        f"skipped={stats['skipped']}, total={stats['total']}[/bold green]"
    )


def load_chunks_file(path: str) -> list[Chunk]:
    from pathlib import Path
    return load_chunks(Path(path))


if __name__ == "__main__":
    main()
