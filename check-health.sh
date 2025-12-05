#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE=deploy/compose/.env.single-a100
nims_ps=$(docker compose --env-file "$ENV_FILE" -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml ps)
rag_ps=$(docker compose --env-file "$ENV_FILE" -f deploy/compose/docker-compose-rag-server.yaml ps)
ingest_ps=$(docker compose --env-file "$ENV_FILE" -f deploy/compose/docker-compose-ingestor-server.yaml ps)

printf "\n=== NIM services ===\n"
printf "%s\n" "$nims_ps"

printf "\n=== RAG server ===\n"
printf "%s\n" "$rag_ps"

printf "\n=== Ingestor ===\n"
printf "%s\n" "$ingest_ps"

printf "\n=== LLM health ===\n"
if docker exec nim-llm-ms curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "nim-llm-ms: READY"
else
  echo "nim-llm-ms: not ready"
fi

printf "\n=== Cache usage ===\n"
du -sh /mnt/nim-cache/.nim-cache 2>/dev/null || true
