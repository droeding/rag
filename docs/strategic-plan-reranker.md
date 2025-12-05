# Strategischer Plan – Reranker-Probleme auf A100DX vGPU

Quelle: Kunde hat tiefe Analyse geliefert (Stand 2025-11-30). Kern: TensorRT-Profil-Selector-Bug in NeMo Retriever Reranker NIM ignoriert Env (`NIM_DISABLE_TRT`, `NIM_PROFILE_INDEX`) und springt auf Profil 3 → OOM → Health 503. Betrifft 1B- und 500M-Reranker.

## Ziel
RAG Blueprint kurzfristig demo-fähig machen, mittelfristig Bug gegenüber NVIDIA adressieren, langfristig volle Reranker-Funktion wiederherstellen.

## Kurzfristig (Demo in Betrieb bringen)
1) Reranker abschalten  
   - In `deploy/compose/docker-compose-rag-server.yaml`: `ENABLE_RERANKER=false`, `RETRIEVER_USE_RERANKER=false`.  
   - Restart nur RAG-Server-Compose.
2) Smoke-Test: Chat, Embedding, OCR, Ingest gegen einen kleinen Datensatz.
3) Klar kommunizieren: Reranking temporär deaktiviert; alle anderen Komponenten laufen.

## Mittelfristig (Workarounds prüfen)
1) ONNX-/Non-TRT-Profil erzwingen  
   - `list-model-profiles` auf dem 500M-Image ausführen, nach ONNX/“non-optimized” Profil-ID suchen.  
   - `NIM_MODEL_PROFILE=<onnx-profile-id>` setzen, Cache-Plan löschen, neu starten.  
   - Health prüfen.
2) Alternativ Cloud-Reranker (NVIDIA API) als Fallback konfigurieren, falls Internet für Demo zulässig.
3) Support-Ticket bei NVIDIA AI Enterprise eröffnen  
   - Inhalte anhängen: aktuelle Logs (Profil-Switch auf 3/1→3 trotz Env), Hinweis A100DX vGPU, ignoriertes `NIM_DISABLE_TRT`.

## Langfristig
1) Auf dedizierte GPU oder zweite vGPU migrieren (kein Host-Reboot aktuell gewünscht; später prüfen).  
2) Sobald NVIDIA Hotfix/Release verfügbar: neues Image testen, Profilwahl validieren.  
3) Optional: eigener leichter Reranker (HuggingFace) in Python/ONNX als Plugin, falls NIM-Fix ausbleibt.

## Offene Punkte / ToDos
- [ ] Reranker deaktivieren und Demo-Skript/Runbook aktualisieren.
- [ ] ONNX-Profil testen und Ergebnis dokumentieren.
- [ ] Support-Ticket erstellen (mit Logauszug aus `docs/reranker-diagnostic.md`).
- [ ] Bei Erfolg mit ONNX: Benchmark (Latenz/Qualität) erfassen.

## Referenzen
- Diagnose: `docs/reranker-diagnostic.md`
- Compose: `deploy/compose/nims.yaml`, `deploy/compose/docker-compose-rag-server.yaml`
