from __future__ import annotations

import os
from collections.abc import Generator

import filelock
import pytest
from sqlalchemy import Engine, create_engine, text
from testcontainers.postgres import PostgresContainer

from memgraph import LocalEmbedder, OpenAIEmbedder, create_graph_schema, create_schema

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
_LOCAL_DIM = 384

_pg_container: PostgresContainer | None = None


@pytest.fixture(scope="session")
def worker_id(request: pytest.FixtureRequest) -> str:
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput.get("workerid", "master")  # type: ignore[union-attr]
    return "master"


@pytest.fixture(scope="session")
def pg(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> Generator[str, None, None]:
    global _pg_container
    root_tmp = (
        tmp_path_factory.getbasetemp().parent
        if worker_id != "master"
        else tmp_path_factory.getbasetemp()
    )
    lock_file = root_tmp / "pg_setup.lock"
    url_file = root_tmp / "pg_url.txt"

    with filelock.FileLock(str(lock_file)):
        if not url_file.exists():
            _pg_container = PostgresContainer(image=POSTGRES_IMAGE)
            _pg_container.start()
            url_file.write_text(_pg_container.get_connection_url())

    yield url_file.read_text().strip()


@pytest.fixture(scope="session")
def engine(pg: str) -> Engine:
    e = create_engine(pg)
    create_schema(e, _LOCAL_DIM)
    create_graph_schema(e, _LOCAL_DIM)
    return e


@pytest.fixture(scope="session")
def local_embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture(scope="session")
def openai_embedder() -> OpenAIEmbedder:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIEmbedder(api_key=key)


@pytest.fixture
def clean_db(engine: Engine) -> Generator[None, None, None]:
    yield
    with engine.connect() as conn:
        conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'entities') THEN
                    TRUNCATE chunks, entities RESTART IDENTITY CASCADE;
                ELSE
                    TRUNCATE chunks RESTART IDENTITY CASCADE;
                END IF;
            END $$;
        """))
        conn.commit()
