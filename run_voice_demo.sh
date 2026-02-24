#!/bin/bash

# Runs fake gateway + orchestrator together for local TTS replies

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start fake gateway in background
"$SCRIPT_DIR/run_fake_gateway.sh" > /tmp/fake_gateway.log 2>&1 &
GATEWAY_PID=$!

# Ensure gateway is stopped on exit
cleanup() {
  kill $GATEWAY_PID 2>/dev/null || true
}
trap cleanup EXIT

# Give gateway a moment to start
sleep 1

echo "✅ Fake gateway started (PID: $GATEWAY_PID)"
echo "   Logs: /tmp/fake_gateway.log"

# Run orchestrator (foreground)
"$SCRIPT_DIR/run_orchestrator.sh" "$@"
