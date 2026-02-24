#!/bin/bash

# Helper script to run the fake gateway test server
# Provides /api/short and WS endpoints for local testing

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate isolated venv if available
if [ -f "$SCRIPT_DIR/.venv311/bin/activate" ]; then
  source "$SCRIPT_DIR/.venv311/bin/activate"
fi

python -m orchestrator.tools.fake_gateway "$@"
