from __future__ import annotations

import logging

from sqlalchemy import Engine, text
from typer.testing import CliRunner

from memgraph.cli import app
from memgraph.ingest import ingest_chunk
from memgraph.llm import MockLLM
from tests.fixtures.canned_llm import MALFORMED, MULTI_ENTITY, SIMPLE_WORLD


def test_reset_empties_data(engine: Engine, local_embedder, clean_db, pg: str) -> None:
    # Seed data
    ingest_chunk(engine, local_embedder, MockLLM([SIMPLE_WORLD]), "The sky is blue.")

    runner = CliRunner()
    result = runner.invoke(app, ["reset", "--yes"], env={"MEMGRAPH_DATABASE_URL": pg})

    assert result.exit_code == 0, result.output
    with engine.connect() as conn:
        for table in ("chunks", "entities", "facts", "fact_entities", "entity_links"):
            count = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()  # noqa: S608
            assert count == 0, f"{table} should be empty after reset, got {count}"


def test_reset_aborts_on_no(engine: Engine, local_embedder, clean_db, pg: str) -> None:
    ingest_chunk(engine, local_embedder, MockLLM([SIMPLE_WORLD]), "The sky is blue.")

    runner = CliRunner()
    result = runner.invoke(app, ["reset"], input="n\n", env={"MEMGRAPH_DATABASE_URL": pg})

    assert result.exit_code != 0
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    assert count > 0, "Data should survive an aborted reset"


def test_debug_flag_emits_pipeline_logs(
    engine: Engine, local_embedder, clean_db, monkeypatch
) -> None:
    monkeypatch.setattr("memgraph.cli._get_components", lambda: (engine, local_embedder))
    monkeypatch.setattr("memgraph.cli._get_llm", lambda: MockLLM([MULTI_ENTITY]))

    runner = CliRunner()
    result = runner.invoke(app, ["--debug", "ingest", "Alice works with Bob"])

    assert result.exit_code == 0, result.output
    # --debug calls logging.basicConfig(force=True) which installs a stderr handler;
    # CliRunner captures stderr in result.output (mix_stderr default). Assert on the
    # captured text rather than caplog, which loses its handler due to force=True.
    output = result.output
    assert "DEBUG memgraph.ingest" in output, f"Expected memgraph.ingest DEBUG in output, got:\n{output}"
    assert "DEBUG memgraph.extract" in output, f"Expected memgraph.extract DEBUG in output, got:\n{output}"
    assert "DEBUG memgraph.resolve" in output, f"Expected memgraph.resolve DEBUG in output, got:\n{output}"


def test_malformed_extraction_emits_warning(
    engine: Engine, local_embedder, clean_db, caplog
) -> None:
    with caplog.at_level(logging.WARNING, logger="memgraph.extract"):
        ingest_chunk(engine, local_embedder, MockLLM([MALFORMED]), "some text to ingest")

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "Failed to parse extraction" in r.message
    ]
    assert len(warnings) >= 1, f"Expected WARNING about failed parse, got: {[r.message for r in caplog.records]}"
