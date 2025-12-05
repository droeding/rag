#!/usr/bin/env bash
# Start the local vLLM-based NIM LLM service with HuggingFace auth.
# Usage:
#   HF_TOKEN=*** ./scripts/start_nim_llm.sh
# Options:
#   ENV_FILE=deploy/compose/.env.single-a100   # override env file
#   SKIP_CHECK=1                               # skip health/model check
#   FOREGROUND=1                               # run in foreground (logs attached)

set -euo pipefail

ENV_FILE=${ENV_FILE:-deploy/compose/.env.single-a100}
SERVICE_NAME=${SERVICE_NAME:-nim-llm}
COMPOSE="docker compose --env-file ${ENV_FILE} -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required (HuggingFace access token for the model)." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: Env file not found at ${ENV_FILE}. Adjust ENV_FILE or create it." >&2
  exit 1
fi

cd "$(dirname "$0")/.."

echo "Starting ${SERVICE_NAME} with compose:"
echo "  ${COMPOSE} up ${FOREGROUND:+}${FOREGROUND:+} ${SERVICE_NAME}"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  HF_TOKEN="${HF_TOKEN}" ${COMPOSE} up ${SERVICE_NAME}
else
  HF_TOKEN="${HF_TOKEN}" ${COMPOSE} up -d ${SERVICE_NAME}
fi

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  echo "Running health/model check..."
  ./scripts/check_llm_model.sh
else
  echo "SKIP_CHECK=1 set; skipping health/model check."
fi

echo "Done. Service: ${SERVICE_NAME}"
