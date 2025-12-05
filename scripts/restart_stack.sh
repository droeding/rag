#!/usr/bin/env bash
# Restart full RAG stack with correct env and service order.
# Usage: optional env overrides before call:
#   NGC_API_KEY=... HF_TOKEN=... ./scripts/restart_stack.sh
#
# Reads defaults from deploy/compose/.env.single-a100

set -euo pipefail

ENV_FILE="${ENV_FILE:-deploy/compose/.env.single-a100}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

# Load compose env defaults (GPU mapping, batch sizes, cache paths)
set -a
. "$ENV_FILE"
set +a

# Require NGC_API_KEY (needed for NIM images)
if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY must be set (export NGC_API_KEY=...)." >&2
  exit 1
fi

COMPOSE_BASE_NIMS=(docker compose --env-file "$ENV_FILE" -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml --profile rag --profile ingestor --profile nemoretriever-ocr)
COMPOSE_RAG=(docker compose --env-file "$ENV_FILE" -f deploy/compose/docker-compose-rag-server.yaml)
COMPOSE_INGEST=(docker compose --env-file "$ENV_FILE" -f deploy/compose/docker-compose-ingestor-server.yaml)
COMPOSE_VDB=(docker compose -f deploy/compose/vectordb.yaml)

echo "Stopping stack..."
"${COMPOSE_RAG[@]}" down || true
"${COMPOSE_INGEST[@]}" down || true
"${COMPOSE_BASE_NIMS[@]}" down || true
"${COMPOSE_VDB[@]}" down || true

echo "Starting Vector DB..."
"${COMPOSE_VDB[@]}" up -d

echo "Starting NIM services (LLM, embeddings, reranker, vision, OCR)..."
"${COMPOSE_BASE_NIMS[@]}" up -d

echo "Starting ingestor..."
"${COMPOSE_INGEST[@]}" up -d

echo "Starting rag-server + frontend..."
"${COMPOSE_RAG[@]}" up -d

echo "Done. Run ./check-health.sh for status."
