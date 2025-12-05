# Local Runbook – 2×GPU Lab Setup (Nemotron‑12B + OCR on GPU1)

This repo is customized for a two‑GPU lab VM (A100/vGPU). Use this runbook to bring the stack up, swap demo collections, and troubleshoot quickly.

## Hardware & OS
- GPUs: 2× A100‑class (80 GB each) or equivalent vGPU; driver ≥ 560; CUDA ≥ 12.4.
- Disk: ≥ 150 GB free (models + Milvus + MinIO + logs). Model cache at `/mnt/nim-cache/.nim-cache`.
- Network: outbound nvcr.io + huggingface.co for first pulls.

## GPU / Service Map
- GPU0: LLM (`nvidia/NVIDIA-Nemotron-Nano-12B-v2`) via vLLM.
- GPU1: Embedding `llama-3.2-nv-embedqa-1b-v2`, Reranker `llama-3.2-nv-rerankqa-1b-v2`, Vision (page/graphic/table), OCR `nemoretriever-ocr-v1`.

## Required env (do **not** commit secrets)
Load `deploy/compose/.env.single-a100` (contains GPU IDs, batch limits, cache paths). Provide at runtime:
- `NGC_API_KEY` (pull NIM images)
- `HF_TOKEN` (pull HF weights for vLLM)
Optional safety knobs (can be exported before starting):
- `NEMORETRIEVER_OCR_BATCH_SIZE=1`
- `NEMORETRIEVER_OCR_CUDA_MEMORY_POOL_MB=2048`

## Start sequence (fresh shell)
```bash
cd /home/demo/projects/rag
set -a; . deploy/compose/.env.single-a100; set +a
# 1) Vector DB
docker compose -f deploy/compose/vectordb.yaml up -d
# 2) NIMs (LLM, embed, rerank, vision, OCR)
docker compose --env-file deploy/compose/.env.single-a100 \
  -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml \
  --profile rag --profile ingestor --profile nemoretriever-ocr up -d
# 3) Ingestor + RAG server
docker compose --env-file deploy/compose/.env.single-a100 \
  -f deploy/compose/docker-compose-ingestor-server.yaml up -d
docker compose --env-file deploy/compose/.env.single-a100 \
  -f deploy/compose/docker-compose-rag-server.yaml up -d
# (Optional) guardrails/observability
```

## Stop sequence
```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml down
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
docker compose --env-file deploy/compose/.env.single-a100 \
  -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml \
  --profile rag --profile ingestor --profile nemoretriever-ocr down
docker compose -f deploy/compose/vectordb.yaml down
```

## Health checks
- Quick bundle: `./check-health.sh`
- LLM: `curl http://nim-llm:8000/v1/models` (use this; `/v1/health/ready` is 404 in vLLM and will show a false red in the UI).
- OCR: `curl http://nemoretriever-ocr:8000/v2/health/ready`
- NV-Ingest: `curl http://nv-ingest-ms-runtime:7670/v1/health/ready` (UI may warn because it probes `/v1` while service exposes `/v2`; functional is OK).
- GPU: `watch -n2 nvidia-smi`

## Ingestion & collection switching
- Put source files under `ingest/`.
- Ingest into a new collection:  
  `python scripts/batch_ingestion.py --folder ingest/ --collection-name <name> --create_collection`
- UI: select collection under “Search collections”. Default collection is intentionally empty.
- To switch demos without wiping data: ingest new collection, select in UI. To fully reset, stop stack, delete `deploy/compose/volumes/{milvus,minio}` and re-ingest.

## Runtime knobs (current defaults)
- `LLM_MAX_TOKENS=4096`, `VECTOR_DB_TOPK=10`, `APP_RETRIEVER_TOPK=5` (see `deploy/compose/docker-compose-rag-server.yaml`).
- Context window of LLM: 128k; keep prompts short or lower `LLM_MAX_TOKENS` if OOM.

## Known quirks
- UI “Summary LLM service not responding” is expected because the health probe hits `/v1/health/ready`; use `/v1/models` to verify readiness.
- NV-Ingest health icon can stay yellow/red due to `/v1` vs `/v2` health endpoints; ingestion still works.
- OCR 500 errors: usually GPU memory pressure. Mitigation: set `NEMORETRIEVER_OCR_BATCH_SIZE=1`, raise `NEMORETRIEVER_OCR_CUDA_MEMORY_POOL_MB` to 2048–4096, then restart NIM stack. Tail logs: `docker logs -f $(docker ps -qf name=ocr)`.

## Housekeeping
- Model cache size visible in `./check-health.sh`; prune old models if disk tight.
- Keep real env files out of git; treat `.env.single-a100` as secret material.
