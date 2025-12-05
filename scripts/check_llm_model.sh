#!/usr/bin/env bash
# Wartet auf nim-llm-ms und zeigt die verfügbaren Modelle an.
# Nutzung: ./scripts/check_llm_model.sh

set -euo pipefail

SERVICE_NAME=${SERVICE_NAME:-nim-llm-ms}
PORT=${PORT:-8000}
TIMEOUT=${TIMEOUT:-600} # Sekunden

echo "Warte auf Container $SERVICE_NAME (Timeout ${TIMEOUT}s)..."

start_ts=$(date +%s)
while true; do
  status=$(docker inspect --format '{{.State.Health.Status}}' "$SERVICE_NAME" 2>/dev/null || echo "missing")
  if [[ "$status" == "healthy" ]]; then
    echo "Container ist healthy."
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > TIMEOUT )); then
    echo "Timeout erreicht, Container nicht healthy. Aktueller Status: $status"
    exit 1
  fi
  sleep 5
done

echo "Frage /v1/models ab..."
if ! docker exec "$SERVICE_NAME" curl -sf "http://localhost:${PORT}/v1/models"; then
  echo "Konnte /v1/models nicht abrufen. Logs (letzte 50 Zeilen):"
  docker logs --tail 50 "$SERVICE_NAME" || true
  exit 1
fi

echo "Done."
