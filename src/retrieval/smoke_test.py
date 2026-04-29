"""
Stage 5: Retriever smoke test CLI.

Requires a running Qdrant instance with the canvas_guides collection populated.

    uv run python -m src.retrieval.smoke_test \\
        --query "Canvas에서 assignment due date와 availability date 차이는?" \\
        --top-k 5
"""
from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from src.retrieval.retriever import get_retriever

console = Console()


@click.command()
@click.option("--query", required=True, help="Query string to search")
@click.option(
    "--qdrant-url", default="http://localhost:6333", show_default=True,
    help="Qdrant server URL",
)
@click.option("--collection", default="canvas_guides", show_default=True)
@click.option("--top-k", default=5, show_default=True, help="Number of results")
@click.option("--role", default=None, help="Filter by role (student/instructor/admin/...)")
@click.option("--category", default=None, help="Filter by category slug")
@click.option(
    "--embedder", "embedder_prefer", default="auto",
    type=click.Choice(["mps", "openai", "auto"]), show_default=True,
)
def main(
    query: str,
    qdrant_url: str,
    collection: str,
    top_k: int,
    role: str | None,
    category: str | None,
    embedder_prefer: str,
) -> None:
    """Smoke test: embed a query and show top-K results from Qdrant."""
    console.print(f"[blue]Query:[/blue] {query}")
    console.print(f"[blue]Qdrant:[/blue] {qdrant_url} / {collection}")

    retriever = get_retriever(
        qdrant_url=qdrant_url,
        collection=collection,
        embedder_prefer=embedder_prefer,
    )
    results = retriever.search(query, top_k=top_k, role=role, category=category)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Role", width=10)
    table.add_column("Title", width=35)
    table.add_column("Text (preview)", width=60)
    table.add_column("Source URL")

    for i, r in enumerate(results, 1):
        preview = r.text[:120].replace("\n", " ") + ("…" if len(r.text) > 120 else "")
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.role or "—",
            (r.title or "—")[:35],
            preview,
            r.source_url,
        )

    console.print(table)
    console.print(f"[green]{len(results)} result(s)[/green]")


if __name__ == "__main__":
    main()
