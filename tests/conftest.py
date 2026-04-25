from __future__ import annotations

import os

import filelock
import pytest
from sqlalchemy import Engine, create_engine, text
from testcontainers.postgres import PostgresContainer

from memgraph import LocalEmbedder, OpenAIEmbedder, create_graph_schema, create_schema

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
_LOCAL_DIM = 384


@pytest.fixture(scope="session")
def worker_id(request: pytest.FixtureRequest) -> str:
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput.get("workerid", "master")  # type: ignore[union-attr]
    return "master"


@pytest.fixture(scope="session")
def pg(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> str:
    root_tmp = (
        tmp_path_factory.getbasetemp().parent
        if worker_id != "master"
        else tmp_path_factory.getbasetemp()
    )
    lock_file = root_tmp / "pg_setup.lock"
    url_file = root_tmp / "pg_url.txt"

    with filelock.FileLock(str(lock_file)):
        if url_file.exists():
            return url_file.read_text().strip()
        container = PostgresContainer(image=POSTGRES_IMAGE)
        container.start()
        url = container.get_connection_url()
        url_file.write_text(url)
        return url


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
def clean_db(engine: Engine) -> object:
    yield
    with engine.connect() as conn:
        # chunks CASCADE → facts → fact_entities; entities CASCADE → entity_links, fact_entities
        conn.execute(text("TRUNCATE chunks, entities RESTART IDENTITY CASCADE"))
        conn.commit()
