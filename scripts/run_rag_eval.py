#!/usr/bin/env python3
"""
Kleines Eval-Skript für den RAG-Server.
Liess Fragen aus JSON, sendet sie an /generate mit Sammlung test5,
optional System-Prompt, schreibt Ergebnisse als JSON.
"""
import argparse
import json
import sys
from pathlib import Path

import requests


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="Pfad zu eval_questions.json")
    ap.add_argument("--system-prompt", help="Pfad zu system_prompt.txt")
    ap.add_argument("--output", required=True, help="Ausgabedatei (JSON)")
    ap.add_argument("--endpoint", default="http://localhost:8081/generate", help="RAG-Server /generate URL")
    ap.add_argument("--collection", default="test5", help="Collection-Name")
    ap.add_argument("--reranker-top-k", type=int, default=5)
    ap.add_argument("--vdb-top-k", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    system_prompt = load_text(Path(args.system_prompt)) if args.system_prompt else ""

    results = []
    for item in questions:
        qid = item.get("id")
        question = item.get("question")
        if not question:
            continue
        payload = {
            "messages": [],
            "collection_names": [args.collection],
            "use_knowledge_base": True,
            "enable_reranker": True,
            "temperature": 0.0,
            "vdb_top_k": args.vdb_top_k,
            "reranker_top_k": args.reranker_top_k,
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": question})

        try:
            r = requests.post(args.endpoint, json=payload, timeout=args.timeout)
            r.raise_for_status()
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "status": r.status_code,
                    "response": r.text,
                }
            )
            print(f"[OK] {qid or question[:40]} -> {len(r.text)} chars")
        except Exception as e:
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "status": "error",
                    "error": str(e),
                }
            )
            print(f"[ERR] {qid or question[:40]} -> {e}", file=sys.stderr)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gespeichert: {args.output}")


if __name__ == "__main__":
    main()

