# Quick Start: MPD-managed Orchestrator

## Prerequisites

Ensure MPD is installed on your system:

```bash
sudo apt update
sudo apt install mpd
```

Verify installation:
```bash
which mpd
mpd --version
```

## How to Run

### Basic Start
```bash
cd /home/stever/projects/openclawstuff/openclaw-voice
python3 orchestrator/main.py
```

Expected output:
```
===================================================
  OpenClaw Voice Orchestrator - Initializing
===================================================

→ Validating runtime configuration...
✓ VAD ready in XXms
...
→ Starting MPD...
✓ MPD ready in XXXms

[Listening for voice input...]
```

### Stop
Press `Ctrl+C` to gracefully shutdown. The orchestrator will:
1. Stop the web service
2. Stop media key detector
3. Stop tool monitor
4. **Stop MPD** ← New!
5. Stop audio capture

## Verification

### Check MPD Status
```bash
mpc status
```

### Check MPD Connection
```bash
nc -zv 127.0.0.1 6600  # Should print: [Connection successful]
```

### Check Process
```bash
ps aux | grep " mpd "
```

## Troubleshooting

### "mpd command not found"
```bash
sudo apt install mpd
```

### "Port 6600 already in use"
Check if another MPD instance is running:
```bash
sudo lsof -i :6600
```

Kill if needed:
```bash
kill <PID>
```

### Music commands not working
1. Verify MPD is running: `mpc status`
2. Check logs in orchestrator output
3. Try playing music directly: `mpc play`
4. Restart orchestrator

## What Changed?

**After**:
- Orchestrator starts MPD as sub-process
- Same orchestrator lifecycle for both
- Cleaner separation of concerns
- Host or container deployable

## For Developers

- **MPD Manager**: `orchestrator/services/mpd_manager.py`
- **Main Integration**: `orchestrator/main.py` (~line 960 and 3365)
- **Tests**: Run `python3 validate_mpd_integration.py`

## Next: Docker Deployment

When you're ready to deploy to Docker/production, see:
- [MPD_ORCHESTRATOR_MANAGEMENT.md](MPD_ORCHESTRATOR_MANAGEMENT.md)
- [MPD_INTEGRATION_SUMMARY.md](MPD_INTEGRATION_SUMMARY.md)
