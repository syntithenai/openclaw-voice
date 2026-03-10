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

uid="$(id -u)"
default_runtime_dir="/run/user/${uid}"
runtime_dir="${XDG_RUNTIME_DIR:-$default_runtime_dir}"
pulse_socket="${runtime_dir}/pulse/native"

selected_service="orchestrator-linux-alsa"
if [[ -S "$pulse_socket" ]]; then
  export XDG_RUNTIME_DIR="$runtime_dir"
  selected_service="orchestrator-linux-pulse"
fi

echo "Selected orchestrator service: ${selected_service}"
if [[ "$selected_service" == "orchestrator-linux-pulse" ]]; then
  echo "PulseAudio/PipeWire socket detected at: ${pulse_socket}"
else
  echo "PulseAudio socket not found; falling back to ALSA (/dev/snd)."
fi

"${COMPOSE[@]}" stop orchestrator orchestrator-linux-alsa orchestrator-linux-pulse >/dev/null 2>&1 || true
"${COMPOSE[@]}" rm -f orchestrator orchestrator-linux-alsa orchestrator-linux-pulse >/dev/null 2>&1 || true

"${COMPOSE[@]}" up -d whisper piper mpd "$selected_service"

echo
echo "Started services: whisper, piper, mpd, ${selected_service}"
echo "Logs: ${COMPOSE[*]} logs -f ${selected_service}"
