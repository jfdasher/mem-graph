#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="memgraph-dev-pg"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running."
else
    docker run --rm -d \
      --name "${CONTAINER_NAME}" \
      -e POSTGRES_PASSWORD=memgraph \
      -e POSTGRES_USER=memgraph \
      -e POSTGRES_DB=memgraph \
      -p 5434:5432 \
      pgvector/pgvector:pg16
    echo "Started ${CONTAINER_NAME}."
fi

echo
echo "Export this before running the CLI:"
echo "  export MEMGRAPH_DATABASE_URL=postgresql://memgraph:memgraph@localhost:5434/memgraph"
