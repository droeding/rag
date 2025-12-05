#!/usr/bin/env bash
set -euo pipefail

# Stop all services of the single-A100 RAG stack in a clean order.
# Requires: docker, docker compose; .env.single-a100 present.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
ENV_FILE="$ROOT/deploy/compose/.env.single-a100"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

cd "$ROOT"

echo "[1/4] Stopping RAG server & UI..."
docker compose -f deploy/compose/docker-compose-rag-server.yaml \
  --env-file "$ENV_FILE" down --remove-orphans

echo "[2/4] Stopping Ingestor stack..."
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml \
  --env-file "$ENV_FILE" down --remove-orphans

echo "[3/4] Stopping NIMs (LLM/Retriever/Vision)..."
docker compose -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml \
  --env-file "$ENV_FILE" --profile a100 --profile retriever down --remove-orphans

echo "[4/4] Stopping Vector DB (Milvus/MinIO/etcd)..."
docker compose -f deploy/compose/vectordb.yaml \
  --env-file "$ENV_FILE" down --remove-orphans

echo "Done. All stack services stopped."
