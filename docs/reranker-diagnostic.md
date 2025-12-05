# Reranker Diagnostic Summary (NIM llama-3.2-nv-rerankqa-1b-v2 on single A100 80GB)

## Environment
- GPU: 1× A100 80GB (GRID A100DX-80C reported by nvidia-smi).
- Disk: root 500 GB (94% belegt), neuer Cache auf `/mnt/nim-cache` (400 GB, >350 GB frei), `MODEL_DIRECTORY=/mnt/nim-cache/.nim-cache`.
- Compose: `deploy/compose/run-single-a100.sh` (vector DB, NIMs, ingestor, RAG).
- API-Key: gesetzt in `deploy/compose/.env.single-a100`.
- Update 2025-11-30 11:21: Zweite A100DX vGPU sichtbar (GPU0+GPU1); Ranking/Embedding auf GPU1 mit ONNX-Profil + TRT disabled starten healthy; Reranker im RAG-Server weiterhin deaktiviert.

## Funktionsfähige Dienste
- LLM: `nvcr.io/nim/meta/llama-3.1-8b-instruct:latest` → healthy.
- Embedding: `llama-3.2-nv-embedqa-1b-v2` → healthy.
- Vision/OCR: page-elements, graphic-elements, table-structure, paddle → healthy.
- NV-Ingest runtime → healthy.
- Milvus/MinIO/Etcd, RAG-Server, Frontend → running.

## Problem (Stand 2025-11-30, abends)
- Alle getesteten Varianten bleiben `unhealthy` (503), inkl. kleiner 500M-Reranker.
- TRT ignoriert Limits und wählt Profil 3 (batch 30, seq 8192, ~23 GB VRAM) → OOM →  
  `Invalid argument: Can not set the specified optimization profile 3[3] ... Expected optimization profile index range 0-6`
  → Triton unload → Health 503.
- Selbst nach Setzen von `NIM_DISABLE_TRT=1` / `NIM_BACKENDS=python` lädt er dennoch TRT.

## Bereits getestete Maßnahmen (alle erfolglos)
1) Health-Fenster erhöht (retries 300, start_period 10m).
2) Profil-Pinning / Limits:
   - `NIM_MODEL_PROFILE=a100x1-trt-fp16-dxtbz8wstg` bzw. `profile_0`
   - `NIM_PROFILE_INDEX=0`
   - `NIM_MAX_BATCH_SIZE=8`, `NIM_MAX_INPUT_LENGTH=2048`
   → Manifest-ID wird erkannt, Laufzeit wechselt trotzdem auf Profil 3.
3) TRT-Plan entfernt:
   ```
   rm -f /mnt/nim-cache/.nim-cache/ngc/hub/models--nim--nvidia--llama-3.2-nv-rerankqa-1b-v2/snapshots/a100x1-trt-fp16-dxtbz8wstg/model.plan
   ```
   → Plan wird neu gezogen, Profil 3 bleibt.
4) TRT abschalten / Python erzwingen (1.8.0):  
   `NIM_DISABLE_TRT=1`, `NIM_FORCE_PYTHON=1`, `NIM_BACKENDS=python`, shm/ipc/ulimits → trotzdem TRT, Profil 3, Crash.
5) Downgrade auf 1.7.0 + kleine Limits → Profil 3, OOM.
6) Kleineres Modell `llama-3.2-nemoretriever-500m-rerank-v2:1.7.0`, Limits (Batch 8, Seq 8192), Profil 0/1 → TRT switcht wieder auf Profil 3, OOM. Auch mit `NIM_DISABLE_TRT=1` wird TRT geladen.
7) Mehrfaches Neustarten.

## Beobachtungen aus Logs
- Verfügbare TRT-Profile: 1/8/16/30 (seq 1024–8192). Selector springt auf Index 3; beim 500M-Modell kurz auf 1, dann auf 3 → OOM.
- 1.7.0-Log (8B): Engine ~2.5 GB, Profilwechsel 3, OOM (19–23 GB).
- 500M-Log: Engine ~1.3 GB, Profil 1 (Batch 8) gewählt, dann Profil 3 → OOM ~23 GB.
- `NIM_MODEL_PROFILE`-Manifest-ID erkannt, trotzdem Profilwechsel.
- `NIM_DISABLE_TRT=1` / `NIM_BACKENDS=python` werden ignoriert (TRT lädt dennoch).

## Abweichung zum Blueprint
- Blueprint erwartet funktionierenden TRT-Reranker auf A100; Profil-Selection-Bug auf A100DX bei 1.7.0/1.8.0 und auch beim 500M-Modell.

## Nächste sinnvolle Schritte (Recherche / Fix-Kandidaten)
1) Reranker temporär deaktivieren: `ENABLE_RERANKER=false`, `RETRIEVER_USE_RERANKER=false` im RAG-Server.
2) ONNX/CPU-Fallback prüfen (Profil-ID mit `list-model-profiles` suchen und setzen, falls vorhanden).
3) NVIDIA-Support-Ticket mit Logs (Profil-Switch auf 3 trotz Env, TRT wird trotz Disable geladen, A100DX vGPU).
4) Alternativ auf dedizierte zweite GPU/vGPU umziehen (vom Nutzer derzeit abgelehnt).

## Aktueller Compose-Zustand (relevante Env für Reranker)
- Image: `nvcr.io/nim/nvidia/llama-3.2-nemoretriever-500m-rerank-v2:1.7.0`
- Env in `deploy/compose/nims.yaml`:
  - `NIM_DISABLE_TRT=1`
  - `NIM_BACKENDS=python`
  - (TRT wird trotzdem geladen → Bug)

## Pfade
- Compose: `deploy/compose/nims.yaml`
- Env: `deploy/compose/.env.single-a100`
- Cache: `/mnt/nim-cache/.nim-cache/ngc/hub/models--nim--nvidia--llama-3.2-nemoretriever-500m-rerank-v2/`

## Kurzanleitung für erneuten Test (Python-Backend)
```bash
cd /home/demo/projects/rag
# Python erzwingen, Plan entfernen, neu starten (500M)
export PLAN=/mnt/nim-cache/.nim-cache/ngc/hub/models--nim--nvidia--llama-3.2-nemoretriever-500m-rerank-v2/snapshots/a100x1-trt-int8-9d9v1pn9nw/model.plan
[ -f "$PLAN" ] && rm -f "$PLAN"
docker compose --env-file deploy/compose/.env.single-a100 \
  -f deploy/compose/nims.yaml -f deploy/compose/nims.a100.yaml up -d nemoretriever-ranking-ms
docker exec nemoretriever-ranking-ms curl -sf http://localhost:8000/v1/health/ready || true
```
