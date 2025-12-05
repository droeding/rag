#!/usr/bin/env bash
# Ingest all files from ingest/dropbox into a collection (default: dropbox).
# Usage: ./scripts/ingest_dropbox.sh [collection_name]
# Requires running stack and Python deps (same as other scripts).

set -euo pipefail

COLLECTION="${1:-dropbox}"
FOLDER="ingest/dropbox"

if [[ ! -d "$FOLDER" ]]; then
  echo "Folder $FOLDER not found. Create it and drop PDFs there."
  exit 1
fi

shopt -s nullglob
files=("$FOLDER"/*)
if (( ${#files[@]} == 0 )); then
  echo "No files in $FOLDER. Add PDFs then re-run."
  exit 0
fi

echo "Ingesting ${#files[@]} file(s) from $FOLDER into collection '$COLLECTION'..."
python3 scripts/batch_ingestion.py --folder "$FOLDER" --collection-name "$COLLECTION" --create_collection
echo "Done."
