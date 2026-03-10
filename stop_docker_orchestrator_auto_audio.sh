#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "ERROR: Neither 'docker compose' nor 'docker-compose' is available."
  exit 1
fi

services=(
  orchestrator
  orchestrator-linux-alsa
  orchestrator-linux-pulse
  whisper
  piper
  mpd
)

"${COMPOSE[@]}" stop "${services[@]}" >/dev/null 2>&1 || true
"${COMPOSE[@]}" rm -f "${services[@]}" >/dev/null 2>&1 || true

echo "Stopped and removed: ${services[*]}"
