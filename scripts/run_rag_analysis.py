#!/usr/bin/env python3
"""
Einfacher One-shot-Runner für den RAG-Server:
- Liest System-Prompt (Pflicht)
- Optional: weiteres Kontext-Dokument als zweite System-Message
- Schickt eine Benutzer-Instruktion (Standard: "Analysiere die Ausschreibungsunterlagen gemäß Systemprompt.")
- Nutzt eine Collection (Default: test5) mit aktiviertem Reranker
- Schreibt das rohe Streaming-Resultat in eine JSON-Datei
"""
import argparse
import json
from pathlib import Path

import requests


def load_text(path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8081/generate", help="RAG-Server /generate URL")
    ap.add_argument("--collection", default="test5", help="Collection-Name")
    ap.add_argument("--system-prompt", required=True, help="Pfad zu system_prompt.md")
    ap.add_argument("--context", help="Optionaler Kontext als zusätzliche System-Message")
    ap.add_argument(
        "--instruction",
        default="Analysiere die Ausschreibungsunterlagen gemäß Systemprompt.",
        help="User-Instruction für den Lauf",
    )
    ap.add_argument("--vdb-top-k", type=int, default=100)
    ap.add_argument("--reranker-top-k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096, help="max_tokens für den LLM-Call")
    ap.add_argument("--no-stream", action="store_true", help="setze stream=False, damit eine einzige JSON-Antwort zurückkommt")
    ap.add_argument("--output", required=True, help="Output-Datei (JSON)")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    messages = [{"role": "system", "content": load_text(args.system_prompt)}]
    if args.context:
        messages.append({"role": "system", "content": load_text(args.context)})
    messages.append({"role": "user", "content": args.instruction})

    payload = {
        "messages": messages,
        "collection_names": [args.collection],
        "use_knowledge_base": True,
        "enable_reranker": True,
        "temperature": args.temperature,
        "vdb_top_k": args.vdb_top_k,
        "reranker_top_k": args.reranker_top_k,
        "max_tokens": args.max_tokens,
        "stream": False if args.no_stream else True,
    }

    r = requests.post(args.endpoint, json=payload, timeout=args.timeout, stream=False)
    r.raise_for_status()

    # Wenn stream=True, kommt ein SSE-ähnlicher Stream mit "data:"-Zeilen. Für Übersicht rekonstruieren wir den Text.
    response_text = ""
    citations = None
    try:
        # Versuch 1: Direkt als JSON (non-stream oder aggregierter Stream)
        obj = json.loads(r.text)
        choices = obj.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
        citations = obj.get("citations")
    except Exception:
        # Versuch 2: SSE-Stream parsen
        try:
            for line in r.text.splitlines():
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]" or not chunk:
                    continue
                obj = json.loads(chunk)
                msg = obj.get("choices", [{}])[0]
                delta = msg.get("delta") or {}
                content = delta.get("content") or ""
                response_text += content
                if "citations" in obj:
                    citations = obj.get("citations")
        except Exception:
            response_text = "(Fehler beim Parsen der Antwort; siehe response_raw)"

    Path(args.output).write_text(
        json.dumps(
            {
                "endpoint": args.endpoint,
                "collection": args.collection,
                "payload": payload,
                "response_raw": r.text,
                "response_text": response_text,
                "citations": citations,
                "status": r.status_code,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Antwort gespeichert in {args.output} (Status {r.status_code}, {len(r.text)} Zeichen)")


if __name__ == "__main__":
    main()
