# mem-graph

Hybrid RAG augmented with LLM-driven fact extraction, entity resolution,
co-occurrence graph, and cross-encoder reranking — all over Postgres.

**Resume bullet**: "Hybrid RAG augmented with LLM-driven fact extraction,
entity resolution, and graph-expansion retrieval, with a cross-encoder
reranker — all over Postgres with a test-first CLI harness."

---

## Setup

```bash
uv sync --extra dev

# Start local Postgres (pgvector, port 5434)
bash scripts/dev-pg.sh
export MEMGRAPH_DATABASE_URL=postgresql://memgraph:memgraph@localhost:5434/memgraph

# Optional: real LLM extraction
export MEMGRAPH_LLM=openai      # or anthropic
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY=...
```

## CLI walkthrough

```bash
# Ingest a conversation transcript (with LLM fact extraction)
memgraph ingest "Alice finished the merger analysis and shared it with Bob."
memgraph ingest "Bob's deadline for the report is next Friday."

# Query — note how graph mode crosses the entity boundary
memgraph facts "Alice" --mode hybrid   # returns Alice facts only
memgraph facts "Alice" --mode graph    # surfaces Bob's deadline via Alice→Bob link
memgraph facts "merger deadline" --mode full --k 5  # hybrid + graph + rerank

# Browse the entity graph
memgraph entities
memgraph entities --name "Alice" --neighbors
```

## Retrieval modes

| Mode | Description |
|---|---|
| `semantic` | pgvector cosine over `facts.embedding` |
| `bm25` | Postgres `ts_rank_cd` over `facts.tsv` |
| `hybrid` | RRF(semantic, bm25) |
| `graph` | Hybrid seeds → entity expansion N hops → rank by `hop_decay × co_count` |
| `full` | RRF(semantic, bm25, graph) → cross-encoder rerank |

**Key demo**: query for "Alice" in graph mode. The system seeds on Alice's
facts, expands through the Alice→Bob co-occurrence link, and retrieves Bob's
deadline fact — even though "Alice" never appears in that fact.

## Tests

```bash
# Mock-only (fast, no API keys required)
uv run pytest -m "not llm" -n auto --timeout=120

# Real-LLM smoke tests (requires API key)
OPENAI_API_KEY=... uv run pytest -m llm -v
ANTHROPIC_API_KEY=... uv run pytest -m llm -v
```
