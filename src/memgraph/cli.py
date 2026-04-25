from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import typer
from sqlalchemy import Engine, create_engine, text

from .embedders import Embedder, LocalEmbedder, OpenAIEmbedder
from .ingest import ingest_chunk, ingest_file
from .llm import LLMProvider
from .retrieve import retrieve_facts
from .schema import create_graph_schema, create_schema

app = typer.Typer(help="memgraph — fact extraction + entity graph + reranking over Postgres")


def _get_components() -> tuple[Engine, Embedder]:
    url = os.environ.get("MEMGRAPH_DATABASE_URL")
    if not url:
        typer.echo("MEMGRAPH_DATABASE_URL is not set.", err=True)
        raise typer.Exit(1)

    embedder_name = os.environ.get("MEMGRAPH_EMBEDDER", "local").lower()
    embedder: Embedder
    if embedder_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            typer.echo("OPENAI_API_KEY is not set.", err=True)
            raise typer.Exit(1)
        embedder = OpenAIEmbedder(api_key=api_key)
    else:
        embedder = LocalEmbedder()

    engine = create_engine(url)
    create_schema(engine, embedder.dimension)
    create_graph_schema(engine, embedder.dimension)
    return engine, embedder


def _get_llm() -> LLMProvider:
    from .llm import AnthropicLLM, MockLLM, OpenAILLM

    provider = os.environ.get("MEMGRAPH_LLM", "mock").lower()
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAILLM(api_key=key)
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicLLM(api_key=key)
    typer.echo("MEMGRAPH_LLM=mock — no facts will be extracted.", err=True)
    return MockLLM([])  # no-op mock for CLI without real LLM


@app.command()
def ingest(
    content: Annotated[str, typer.Argument(help="Text to ingest, or path to a file/directory")],
    doc_id: Annotated[str | None, typer.Option("--doc-id")] = None,
    tag: Annotated[list[str], typer.Option("--tag")] = [],  # noqa: B006
    chunk_size: Annotated[int, typer.Option("--chunk-size")] = 512,
) -> None:
    """Ingest text or a file/directory. Set MEMGRAPH_LLM=openai|anthropic for extraction."""
    engine, embedder = _get_components()
    llm = _get_llm()
    path = Path(content)
    tag_list = list(tag) or None
    if path.exists():
        ids = ingest_file(engine, embedder, llm, path, chunk_size=chunk_size, tags=tag_list)
        typer.echo(f"Ingested {len(ids)} chunks from {content}.")
    else:
        uid = ingest_chunk(engine, embedder, llm, content, document_id=doc_id, tags=tag_list)
        typer.echo(str(uid))


@app.command()
def facts(
    query: Annotated[str, typer.Argument(help="Query string")],
    k: Annotated[int, typer.Option("--k")] = 10,
    mode: Annotated[
        Literal["semantic", "bm25", "hybrid", "graph", "full"],
        typer.Option("--mode"),
    ] = "hybrid",
    hops: Annotated[int, typer.Option("--hops")] = 1,
) -> None:
    """Query facts. --mode: semantic|bm25|hybrid|graph|full"""
    engine, embedder = _get_components()
    results = retrieve_facts(engine, embedder, query, mode=mode, k=k, hops=hops)
    if not results:
        typer.echo("No results.")
        return
    for fact in results:
        snippet = fact.text[:120].replace("\n", " ")
        typer.echo(f"[{fact.score:.4f}] [{fact.fact_type}] {snippet}")


@app.command()
def entities(
    name: Annotated[str | None, typer.Option("--name")] = None,
    show_neighbors: Annotated[bool, typer.Option("--neighbors")] = False,
) -> None:
    """List entities. --name filters by name fragment; --neighbors shows links."""
    engine, _ = _get_components()
    with engine.connect() as conn:
        if name:
            rows = conn.execute(
                text(
                    "SELECT id, name, type, aliases FROM entities "
                    "WHERE name ILIKE :pat ORDER BY name LIMIT 50"
                ),
                {"pat": f"%{name}%"},
            ).fetchall()
        else:
            rows = conn.execute(
                text("SELECT id, name, type, aliases FROM entities ORDER BY name LIMIT 50")
            ).fetchall()

    for row in rows:
        typer.echo(f"{row.id}  {row.name}  [{row.type}]  aliases={list(row.aliases or [])}")
        if show_neighbors:
            with engine.connect() as conn:
                links = conn.execute(
                    text("""
                        SELECT
                            e.name AS neighbor_name,
                            el.co_count
                        FROM entity_links el
                        JOIN entities e ON e.id = CASE
                            WHEN el.entity_a_id = :id THEN el.entity_b_id
                            ELSE el.entity_a_id
                        END
                        WHERE el.entity_a_id = :id OR el.entity_b_id = :id
                        ORDER BY el.co_count DESC
                        LIMIT 10
                    """),
                    {"id": str(row.id)},
                ).fetchall()
            for lnk in links:
                typer.echo(f"    → {lnk.neighbor_name} (co_count={lnk.co_count})")
