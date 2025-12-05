#!/usr/bin/env bash
set -euo pipefail

# Simple health watcher for the RAG/NIM stack.
# Usage: ./scripts/watch-rag-health.sh
# Optional: INTERVAL=5 ./scripts/watch-rag-health.sh

INTERVAL=${INTERVAL:-5}

containers=(
  nim-llm-ms
  nemoretriever-embedding-ms
  nemoretriever-ranking-ms
  rag-server
  rag-frontend
  milvus-standalone
  milvus-minio
  milvus-etcd
  compose-redis-1
  compose-nv-ingest-ms-runtime-1
  ingestor-server
)

while true; do
  echo "=== $(date -Iseconds) ==="
  for c in "${containers[@]}"; do
    status=$(docker inspect --format='{{.State.Health.Status}}' "$c" 2>/dev/null || docker inspect --format='{{.State.Status}}' "$c" 2>/dev/null || echo "missing")
    echo "$c: $status"
  done
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo
  sleep "$INTERVAL"
done
