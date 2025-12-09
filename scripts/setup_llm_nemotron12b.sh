#!/usr/bin/env bash
# Pull und Start für das vLLM-OpenAI-Serving mit nvidia/NVIDIA-Nemotron-Nano-12B-v2
# Nutzung:
#   HF_TOKEN=<dein_hf_token> ./scripts/setup_llm_nemotron12b.sh
#
# Schritte:
# 1) docker compose up nim-llm im Vordergrund (zum vollständigen Pull)
# 2) Ctrl+C wenn "Pull complete" durch ist
# 3) Start im Hintergrund (detached)
# 4) Health/Models prüfen

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.huggingface/token" ]]; then
  HF_TOKEN=$(cat "$HOME/.huggingface/token")
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Bitte HF_TOKEN setzen (HuggingFace Access Token)."
  echo "Tipp: echo 'hf_...' > ~/.huggingface/token"
  exit 1
fi

cd "$(dirname "$0")/.."

COMPOSE="docker compose --env-file deploy/compose/.env.single-a100 -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml"

echo "=== Schritt 1: Image/Service im Vordergrund ziehen (Ctrl+C nach Pull complete) ==="
echo "Command: HF_TOKEN=*** $COMPOSE up nim-llm"
HF_TOKEN="${HF_TOKEN}" $COMPOSE up nim-llm

echo "=== Schritt 2: Neustart im Hintergrund ==="
HF_TOKEN="${HF_TOKEN}" $COMPOSE up -d nim-llm

echo "=== Schritt 3: Health/Models prüfen ==="
./scripts/check_llm_model.sh

echo "Fertig. Wenn healthy, Benchmark laufen lassen mit run_rag_analysis."
