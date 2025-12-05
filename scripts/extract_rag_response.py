#!/usr/bin/env python3
"""
Extracts `response_text` from a run_rag_analysis JSON and writes it to a .md (or .txt) file.
Usage:
  python scripts/extract_rag_response.py \
    --input tender-analyse/results/analysis_8b_long.json \
    --output tender-analyse/results/analysis_8b_long.md
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON from run_rag_analysis")
    ap.add_argument("--output", required=True, help="Output markdown/text path")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    text = data.get("response_text") or data.get("response_raw") or ""
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} chars to {args.output}")


if __name__ == "__main__":
    main()
