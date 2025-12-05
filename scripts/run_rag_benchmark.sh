#!/usr/bin/env bash
# Simple wrapper to run RAG analysis with current model or (optionally) switch to a large model.
# Usage:
#   ./scripts/run_rag_benchmark.sh --output tender-analyse/results/analysis_8b.json
#   ./scripts/run_rag_benchmark.sh --output tender-analyse/results/analysis_49b.json
# Flags:
#   --collection   Name of the collection (default: test5)
#   --system       Path to system prompt (default: tender-analyse/prompts/system_prompt.md)
#   --context      Path to context prompt (default: tender-analyse/prompts/context_bechtle.md)
#   --instruction  User instruction (default provided)
#   --endpoint     RAG server endpoint (default: http://localhost:8081/generate)
#   --vdb-top-k    (default 100)
#   --reranker-top-k (default 10)
set -euo pipefail

collection="test5"
system_prompt="tender-analyse/prompts/system_prompt.md"
context_prompt="tender-analyse/prompts/context_bechtle.md"
instruction="Analysiere die Ausschreibungsunterlagen gemäß Systemprompt."
endpoint="http://localhost:8081/generate"
vdb_top_k=100
reranker_top_k=10
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection) collection="$2"; shift 2 ;;
    --system) system_prompt="$2"; shift 2 ;;
    --context) context_prompt="$2"; shift 2 ;;
    --instruction) instruction="$2"; shift 2 ;;
    --endpoint) endpoint="$2"; shift 2 ;;
    --vdb-top-k) vdb_top_k="$2"; shift 2 ;;
    --reranker-top-k) reranker_top_k="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$output" ]]; then
  echo "Bitte --output angeben (z.B. tender-analyse/results/analysis_8b.json)"; exit 1;
fi

python scripts/run_rag_analysis.py \
  --collection "$collection" \
  --system-prompt "$system_prompt" \
  --context "$context_prompt" \
  --instruction "$instruction" \
  --endpoint "$endpoint" \
  --vdb-top-k "$vdb_top_k" \
  --reranker-top-k "$reranker_top_k" \
  --output "$output"

echo "Fertig: $output"

