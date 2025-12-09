#!/usr/bin/env bash
# Manage full RAG stack: start | stop | restart | status
# Usage:
#   ./scripts/stack.sh start
#   ./scripts/stack.sh stop
#   ./scripts/stack.sh restart
#   ./scripts/stack.sh status
#
# Optional: set ENV_FILE to override compose env (default: deploy/compose/.env.single-a100)
# Required: NGC_API_KEY must be set (env or inside ENV_FILE). HF_TOKEN only needed for first vLLM pull.

set -euo pipefail

ACTION="${1:-}"
ENV_FILE="${ENV_FILE:-deploy/compose/.env.single-a100}"

COMPOSE_NIMS=(docker compose --env-file "$ENV_FILE" -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml --profile rag --profile ingestor --profile nemoretriever-ocr)
COMPOSE_RAG=(docker compose --env-file "$ENV_FILE" -f deploy/compose/docker-compose-rag-server.yaml)
COMPOSE_INGEST=(docker compose --env-file "$ENV_FILE" -f deploy/compose/docker-compose-ingestor-server.yaml)
COMPOSE_VDB=(docker compose -f deploy/compose/vectordb.yaml)

require_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
  fi
  # Load env defaults (GPU mapping, cache paths, batch sizes). Missing keys are allowed.
  set -a
  . "$ENV_FILE"
  set +a

  # Auto-load HF token if not provided
  if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.huggingface/token" ]]; then
    HF_TOKEN=$(cat "$HOME/.huggingface/token")
  fi
}

stop_stack() {
  echo "Stopping rag-server/frontend..."
  "${COMPOSE_RAG[@]}" down --remove-orphans || true
  echo "Stopping ingestor..."
  "${COMPOSE_INGEST[@]}" down --remove-orphans || true
  echo "Stopping NIM services (all, no profile filter)..."
  docker compose --env-file "$ENV_FILE" -f deploy/compose/nims.yaml down --remove-orphans || true
  echo "Stopping vector DB..."
  "${COMPOSE_VDB[@]}" down --remove-orphans || true
  echo "Removing network nvidia-rag if present..."
  docker network rm nvidia-rag >/dev/null 2>&1 || true
}

start_stack() {
  require_env
  echo "Starting vector DB..."
  "${COMPOSE_VDB[@]}" up -d
  echo "Starting NIM services (LLM, embeddings, reranker, vision, OCR)..."
  "${COMPOSE_NIMS[@]}" up -d
  echo "Starting ingestor..."
  "${COMPOSE_INGEST[@]}" up -d
  echo "Starting rag-server + frontend..."
  "${COMPOSE_RAG[@]}" up -d
}

status_stack() {
  if command -v ./check-health.sh >/dev/null 2>&1; then
    ./check-health.sh
  else
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
  fi
}

case "$ACTION" in
  start)
    start_stack
    ;;
  stop)
    stop_stack
    ;;
  restart|"")
    stop_stack
    start_stack
    ;;
  status)
    status_stack
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac

echo "Done."
