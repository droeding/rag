# NVIDIA RAG Blueprint - Projekt Status

> Zuletzt aktualisiert: 2025-12-09

## Projekt-Übersicht

Dieses Projekt ist ein Fork/Anpassung des **NVIDIA RAG Blueprint** - ein Retrieval-Augmented Generation System für Enterprise-Anwendungen.

**Repository:** `github.com/droeding/rag`  
**Branch:** `backup/adapted-2025-12-05`  
**Basis:** NVIDIA RAG Blueprint v2.3.0.dev

---

## Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React/TypeScript)                  │
│                    Port: 3000 (rag-frontend)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  RAG Server   │    │Ingestor Server│    │   Milvus DB   │
│  Port: 8081   │    │   Port: 8082  │    │  Port: 19530  │
│  (FastAPI)    │    │   (FastAPI)   │    │   + MinIO     │
└───────┬───────┘    └───────┬───────┘    └───────────────┘
        │                    │
        ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NIM Microservices (GPU)                      │
├─────────────────────────────────────────────────────────────────┤
│  GPU 0:                                                         │
│    • LLM: NVIDIA-Nemotron-Nano-12B-v2 (vLLM) - Port 8999        │
│                                                                 │
│  GPU 1:                                                         │
│    • Embedding: llama-3.2-nv-embedqa-1b-v2 - Port 9080          │
│    • Reranker: llama-3.2-nv-rerankqa-1b-v2 - Port 1976          │
│    • Vision: page-elements, graphic-elements, table-structure   │
│    • OCR: nemoretriever-ocr-v1 - Port 8012                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Anpassungen gegenüber NVIDIA Blueprint

| Bereich | Original NVIDIA | Diese Anpassung |
|---------|-----------------|-----------------|
| **LLM Modell** | Nemotron-49B (NIM) | Nemotron-Nano-12B-v2 (vLLM) |
| **Kontext-Länge** | Standard | 131.072 Token |
| **GPU-Verteilung** | Flexibel | GPU0=LLM, GPU1=Rest |
| **OCR** | PaddleOCR | nemoretriever-ocr-v1 |
| **Model Cache** | `/mnt/nim-cache` | `$HOME/.cache/model-cache` |
| **HF Token** | Manuell setzen | Auto-Load von `~/.huggingface/token` |

---

## Tech Stack

### Backend (Python 3.12+)
- **Framework:** FastAPI + Uvicorn
- **RAG Orchestration:** LangChain
- **Vector DB:** Milvus (mit cuVS GPU-Beschleunigung)
- **Object Storage:** MinIO
- **Cache:** Redis
- **Package Manager:** UV

### Frontend (TypeScript)
- **Framework:** React 18 + Vite
- **State Management:** Zustand
- **UI:** Custom Components + KUI
- **Testing:** Vitest

### Infrastructure
- **Container:** Docker Compose
- **GPU:** NVIDIA (CUDA)
- **Observability:** OpenTelemetry + Prometheus

---

## Wichtige Dateien und Verzeichnisse

### Konfiguration
```
deploy/compose/
├── .env.single-a100          # GPU-Mapping, API Keys, Model Cache
├── nims.yaml                 # NIM Microservices Definition
├── docker-compose-*.yaml     # Service Compose Files
└── vectordb/milvus/          # Milvus Konfiguration
```

### Backend Quellcode
```
src/nvidia_rag/
├── rag_server/               # RAG API Server
│   ├── server.py             # FastAPI Endpoints
│   ├── main.py               # NvidiaRAG Klasse
│   └── prompt.yaml           # System Prompts
├── ingestor_server/          # Dokument-Ingestion Server
│   ├── server.py             # FastAPI Endpoints
│   └── nvingest.py           # NV-Ingest Integration
└── utils/                    # Shared Utilities
    ├── vdb/                  # Vector DB Abstraktionen
    ├── embedding.py          # Embedding Service
    ├── reranker.py           # Reranker Service
    └── llm.py                # LLM Service
```

### Frontend Quellcode
```
frontend/src/
├── api/                      # API Hooks
├── components/               # React Components
├── hooks/                    # Custom Hooks
├── pages/                    # Page Components
├── store/                    # Zustand Stores
└── types/                    # TypeScript Types
```

---

## Entwicklungs-Setup

### Voraussetzungen
- Python 3.12+
- Node.js 18+ (für Frontend)
- Docker + Docker Compose
- NVIDIA GPU mit CUDA
- NGC API Key
- HuggingFace Token (für bestimmte Modelle)

### HuggingFace Token einrichten
```bash
# Token speichern (wird automatisch geladen)
echo 'hf_YOUR_TOKEN' > ~/.huggingface/token
```

### Stack starten
```bash
# Vollständigen Stack starten
./scripts/stack.sh start

# Nur LLM starten
./scripts/start_nim_llm.sh

# Stack stoppen
./scripts/stack.sh stop
```

### Health Check
```bash
./check-health.sh
# oder
curl http://localhost:8081/v1/health?check_dependencies=true
```

---

## API Endpoints

### RAG Server (Port 8081)
| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/v1/health` | GET | Health Check |
| `/v1/generate` | POST | RAG Antwort generieren |
| `/v1/chat/completions` | POST | OpenAI-kompatibel |
| `/v1/search` | POST | Dokument-Suche |
| `/v1/summary` | GET | Dokument-Zusammenfassung |

### Ingestor Server (Port 8082)
| Endpoint | Methode | Beschreibung |
|----------|---------|--------------|
| `/v1/health` | GET | Health Check |
| `/v1/documents` | POST | Dokumente hochladen |
| `/v1/documents` | GET | Dokumente auflisten |
| `/v1/documents` | DELETE | Dokumente löschen |
| `/v1/collections` | GET/POST/DELETE | Collections verwalten |

---

## Bekannte Einschränkungen / Offene Punkte

1. **UI Health-Warnung:** NV-Ingest zeigt manchmal "unhealthy" wegen Healthcheck-Endpunkt Inkonsistenz (funktional OK)
2. **GPU Memory:** GPU1 hostet mehrere Services - bei großen Dokumenten ggf. Memory-Engpass
3. **Token Limits:** LLM_MAX_TOKENS=4096, VECTOR_DB_TOPK=10, APP_RETRIEVER_TOPK=5

---

## Nützliche Befehle

```bash
# GPU Monitoring
watch -n2 nvidia-smi

# Container Status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Logs eines Services
docker logs -f nim-llm-ms

# In Container einsteigen
docker exec -it rag-server bash

# Tests ausführen (Backend)
cd tests/unit && pytest

# Tests ausführen (Frontend)
cd frontend && pnpm test
```

---

## Weiterentwicklung

### Ideen / Backlog
- [ ] Multi-GPU Skalierung für größere Modelle
- [ ] Custom Prompt Templates per Collection
- [ ] Batch-Ingestion Optimierung
- [ ] Frontend: Conversation Export
- [ ] Evaluation Pipeline (RAGAS)

### Änderungshistorie
| Datum | Beschreibung |
|-------|--------------|
| 2025-12-09 | HF-Token Auto-Loading, OCR Config Fix |
| 2025-12-05 | LLM auf Nemotron-12B umgestellt, vLLM statt NIM |
| 2025-11-30 | Initiale Anpassung des Blueprints |

---

## Referenzen

- [NVIDIA RAG Blueprint Docs](https://github.com/NVIDIA-AI-Blueprints/rag)
- [LangChain Docs](https://python.langchain.com/)
- [Milvus Docs](https://milvus.io/docs)
- [vLLM Docs](https://docs.vllm.ai/)
