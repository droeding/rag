# Tender-Analyse Benchmark

Ziel: Prompts/Fragen gegen den hochgeladenen Vergabe-Datensatz (`test5`) testen – erst mit dem aktuellen 8B-LLM, später mit dem großen Modell – und Ergebnisse vergleichbar ablegen.

## Struktur
- `prompts/system_prompt.md` – dein System-Prompt (aus Gemini übernommen).
- `prompts/context_bechtle.md` – Kontextdokument, das im Prompt genutzt werden kann.
- `docs/` – Quell-PDFs:
  - `ohb_rfi_strategic_it_partner.pdf` (RFI)
  - `ohb_analysis_strategic_it_partner.pdf` (Ergebnis OHB)
  - `ovgu_ism_leistungsbeschreibung.pdf` (Ausschreibung/Leistungsbeschreibung)
  - `ovgu_analysis_ism.pdf` (Ergebnis OVGU)
- `eval_questions.sample.json` – Vorlage für Testfragen inkl. Soll-Fakten.
- `results/` – Output-Ablage (z. B. `results_8b.json`, `results_49b.json`).

## Workflow
### Ein-Prompt-Analyse (aktuelles Szenario)
Nutze `scripts/run_rag_analysis.py`, um System-Prompt (+ optional Kontext) einmal durch den RAG-Server zu schicken:
```bash
python scripts/run_rag_analysis.py \
  --collection test5 \
  --system-prompt tender-analyse/prompts/system_prompt.md \
  --context tender-analyse/prompts/context_bechtle.md \
  --instruction "Analysiere die Ausschreibungsunterlagen gemäß Systemprompt." \
  --output tender-analyse/results/analysis_8b.json
```
- Reranker aktiv; Defaults: `vdb_top_k=100`, `reranker_top_k=10`, Endpoint `http://localhost:8081/generate` (per Flag änderbar).

### Vergleich 8B vs. großes Modell
1) Lauf wie oben (8B) -> `results/analysis_8b.json`.
2) In `deploy/compose/docker-compose-rag-server.yaml` die LLM-Variablen auf das große Modell setzen, RAG-Server neu starten.
3) Script erneut ausführen, z. B.:
```bash
python scripts/run_rag_analysis.py \
  --collection test5 \
  --system-prompt tender-analyse/prompts/system_prompt.md \
  --context tender-analyse/prompts/context_bechtle.md \
  --instruction "Analysiere die Ausschreibungsunterlagen gemäß Systemprompt." \
  --output tender-analyse/results/analysis_49b.json
```
4) Outputs vergleichen (Inhalt, Länge, Halluzinationen, Zitate).

### Optional: Mehrere Fragen
- `scripts/run_rag_eval.py` + eigenes `eval_questions.json` nutzen (Vorlage: `eval_questions.sample.json`).

## Parameter
`run_rag_analysis.py` Default-Endpoint: `http://localhost:8081/generate`, Collection: `test5`, Reranker aktiv.
