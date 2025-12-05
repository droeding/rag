#!/usr/bin/env bash
set -euo pipefail

# One-shot startup for single A100 (80GB) with local 8B LLM.

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENV_FILE="$ROOT/deploy/compose/.env.single-a100"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file missing: $ENV_FILE" >&2
  exit 1
fi

export USERID=${USERID:-$(id -u)}
export DOCKER_BUILDKIT=0

compose() {
  docker compose --env-file "$ENV_FILE" "$@"
}

cd "$ROOT"

echo "[1/4] Vector DB (Milvus/Minio/Etcd)"
compose -f deploy/compose/vectordb.yaml up -d

echo "[2/5] NIMs (LLM 8B + Retriever + Vision/OCR)"
# Start all NIM services so nv-ingest finds paddle/page-elements/graphic-elements/table-structure
compose -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml up -d

echo "[3/5] Ingestor stack"
compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d

echo "[4/5] RAG server"
compose -f deploy/compose/docker-compose-rag-server.yaml up -d

echo "[5/5] Status"
compose -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml ps
compose -f deploy/compose/docker-compose-ingestor-server.yaml ps
compose -f deploy/compose/vectordb.yaml ps
compose -f deploy/compose/docker-compose-rag-server.yaml ps
