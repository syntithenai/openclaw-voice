#!/bin/bash

# Helper script to run the OpenClaw Voice Orchestrator
# Activates isolated Python 3.11 venv and runs orchestrator

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate isolated venv
source "$SCRIPT_DIR/.venv311/bin/activate"

# Run orchestrator with any passed arguments
python -m orchestrator.main "$@"
