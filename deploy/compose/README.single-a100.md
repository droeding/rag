# Single A100 (80GB) Local Deployment Runbook

## Preconditions
- GPU: 1× A100 80GB, driver >= 560, CUDA >= 12.9.
- Disk frei: >= 130GB empfohlen (Modelle + VDB + Logs).
- NVIDIA API Key: `NVIDIA_API_KEY` (gleich `NGC_API_KEY`).

## Einmalig
```bash
mkdir -p ~/.cache/nim vectordb ingest
```

## Environment laden
```bash
set -a
. deploy/compose/.env.single-a100   # enthält GPU=0, kleines 8B-LLM, konservative Batch-Limits
set +a
```

## Docker Login (für NIM Pull)
```bash
echo "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

## Dienste starten (Reihenfolge)
```bash
# 1) Vector DB (Milvus GPU-Index hier deaktiviert; GPU-Pool per Env begrenzt)
docker compose -f deploy/compose/vectordb.yaml up -d

# 2) NIMs mit 8B-LLM Override
docker compose -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml \
  --profile a100 --profile retriever up -d

# 3) Ingestor + RAG Server
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d

# 4) (Optional) Guardrails/Observability/UI
# docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml up -d
# docker compose -f deploy/compose/observability.yaml up -d
# frontend siehe docs/user-interface.md
```

## Ingestion Beispiel
```bash
python scripts/batch_ingestion.py --folder ingest/ --collection-name my_collection --create_collection
```

## Smoke-Test
```bash
python scripts/retriever_api_usage.py "Was ist RAG?"
```

## Stoppen
```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml down
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
docker compose -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml --profile a100 --profile retriever down
docker compose -f deploy/compose/vectordb.yaml down
```

## Hinweise
- Milvus-GPU-Index ist standardmäßig aus (`APP_VECTORSTORE_ENABLEGPUSEARCH/INDEX=False`). Bei Bedarf einschalten und `KNOWHERE_GPU_MEM_POOL_SIZE` in der Env anpassen.
- Batch-Limits für Embedding/Reranker in `.env.single-a100` gesetzt, um OOM zu vermeiden.
- Modell-Cache unter `~/.cache/nim` (kann auf anderes Volume gelegt werden, `MODEL_DIRECTORY` anpassen).
