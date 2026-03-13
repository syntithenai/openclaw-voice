# MPD Orchestrator Lifecycle Management - Implementation Summary

## Overview

This update transfers MPD (Music Player Daemon) lifecycle management from docker-compose to the orchestrator. The orchestrator now:
- **Starts MPD** during initialization (after config validation, before main loop)
- **Stops MPD** during graceful shutdown
- **Monitors MPD readiness** with socket connection validation

## Changes Made

### 1. New Module: `orchestrator/services/`

**File**: `orchestrator/services/__init__.py`
- Package marker for services collection

**File**: `orchestrator/services/mpd_manager.py` (NEW)
- **Class**: `MPDManager` - Manages MPD process lifecycle
- **Methods**:
  - `__init()` - Initialize with port, host, optional config path
  - `_find_mpd_config()` - Searches standard locations for mpd.conf
  - `start()` - Start MPD process via subprocess.Popen()
  - `stop()` - Graceful SIGTERM shutdown with SIGKILL fallback
  - `restart()` - Stop and restart
  - `is_running()` - Check process status
  - `get_pid()` - Get current PID
  - `wait_for_ready()` - Socket connection test with timeout
  - `cleanup()` - Stop process and clean resources
- **Features**:
  - Auto-discovery of mpd.conf in standard locations
  - Graceful shutdown with 5-second timeout
  - Ready-state verification via TCP connection
  - Detailed logging and error messages
  - Helpful installation guidance

### 2. Updated: `orchestrator/main.py`

**Import** (line ~62):
```python
from orchestrator.services.mpd_manager import MPDManager
```

**Startup** (line ~960):
```python
# MPD (Music Player Daemon)
print("→ Starting MPD...", flush=True)
mpd_manager = MPDManager(mpd_port=6600, mpd_host="127.0.0.1")
mpd_start = time.monotonic()
if mpd_manager.start():
    if mpd_manager.wait_for_ready(timeout_sec=5):
        mpd_elapsed = int((time.monotonic() - mpd_start) * 1000)
        logger.info("✓ MPD ready in %dms", mpd_elapsed)
        print(f"✓ MPD ready in {mpd_elapsed}ms", flush=True)
    else:
        logger.warning("MPD started but did not become ready within timeout")
else:
    logger.warning("Failed to start MPD; music features may be unavailable")
    mpd_manager = None
```

**Shutdown** (line ~3365):
```python
# Cleanup MPD if running
if mpd_manager:
    logger.info("Stopping MPD...")
    mpd_manager.cleanup()
```

### 3. Updated: `docker-compose.yml`

**Changes**:
- **Removed** `- mpd` from orchestrator's `depends_on` list
- **Removed** the standalone `mpd` service definition entirely
- Updated orchestrator container MPD connectivity to `127.0.0.1`

**Before**:
```yaml
depends_on:
  - whisper
  - piper
  - mpd
```

**After**:
```yaml
depends_on:
  - whisper
  - piper
```

### 4. New Documentation

**File**: `MPD_ORCHESTRATOR_MANAGEMENT.md`
- Architecture explanation (before/after)
- Installation requirements (apt install mpd)
- Configuration file locations and examples
- Running the orchestrator (host vs docker)
- Troubleshooting guide
- Integration points in code
- Revert instructions

**File**: `validate_mpd_integration.py`
- Validation script for syntax and imports
- Can be run to verify implementation

## Startup Sequence

```
OpenClaw Voice Orchestrator - Initializing
→ Validating runtime configuration...
✓ VAD loaded in Xms
→ Loading wake word detector...
✓ Wake word detector loaded in Xms
→ Emotion detection model...
✓ SenseVoice loaded in Xms
→ Initializing gateway...
✓ Gateway ready in Xms
→ Starting MPD...
✓ MPD ready in Xms
[Main loop begins]
```

## Shutdown Sequence

```
[Main loop interrupted]
Stopping embedded web UI service...
Stopping media key detector...
Stopping tool monitor...
Stopping MPD...
[Clean exit]
```

## Requirements

### System
- **Linux/macOS**: `mpd` package installed
- **Installation**: `sudo apt install mpd` (Debian/Ubuntu)

### Python
- No new external dependencies
- Uses stdlib: `subprocess`, `socket`, `time`, `logging`, `pathlib`

### Environment
- MPD listens on `127.0.0.1:6600` (configurable)
- Music directory: `~/Music` (or configured in mpd.conf)
- State directory: `~/.config/mpd` (standard location)

## Configuration Discovery

MPD manager searches for configuration in this order:
1. `OPENCLAW_MPD_CONFIG` (explicit override)
2. `~/.config/mpd/mpd.conf` (user level)
3. `orchestrator/services/mpd.conf` (bundled config)
4. `/etc/mpd.conf` (system level)
5. Default MPD configuration

## Running the Orchestrator

### Host-based (Recommended)
```bash
cd /home/stever/projects/openclawstuff/openclaw-voice
python3 orchestrator/main.py
```

### Container-based (Adjusted workflow)
If running orchestrator in container:
- MPD is installed directly in the orchestrator container image
- The orchestrator starts MPD locally and connects on `127.0.0.1:6600`
- See `MPD_ORCHESTRATOR_MANAGEMENT.md` for instructions

## Benefits

✅ **Single control point** - Orchestrator manages full lifecycle
✅ **Graceful shutdown** - Proper SIGTERM handling
✅ **Failure recovery** - Can restart MPD if needed
✅ **Health monitoring** - Ready-state validation
✅ **Simplified docker-compose** - Less service interdependency
✅ **Flexible deployment** - Works on host or in container
✅ **Better logging** - Explicit startup/shutdown traces

## Rollback

There is no alternate containerized MPD mode to roll back to.
If you need a custom MPD configuration, set `OPENCLAW_MPD_CONFIG` or override `~/.config/mpd/mpd.conf`.

## Testing

### Verify MPD Installation
```bash
which mpd
mpd --version
```

### Test MPD Manager
```bash
cd /home/stever/projects/openclawstuff/openclaw-voice
python3 validate_mpd_integration.py
```

### Test Orchestrator Startup
```bash
python3 orchestrator/main.py
# Watch for: "✓ MPD ready in Xms"
```

### Test Music Features
```bash
# In another terminal with orchestrator running
mpc status
mpc play
mpc pause
```

## Files Modified

| File | Change Type | Reason |
|------|------------|--------|
| `orchestrator/services/__init__.py` | NEW | Service package |
| `orchestrator/services/mpd_manager.py` | NEW | MPD lifecycle class |
| `orchestrator/services/mpd.conf` | NEW | Bundled MPD config for container runs |
| `orchestrator/main.py` | MODIFIED | Import + startup/shutdown |
| `docker-compose.yml` | MODIFIED | Remove legacy compose MPD entry |
| `MPD_ORCHESTRATOR_MANAGEMENT.md` | NEW | User documentation |
| `validate_mpd_integration.py` | NEW | Integration validation |

## Integration Points

### Startup
- File: `orchestrator/main.py`
- Function: `run_orchestrator()`
- Line: ~960 (after gateway initialization)
- Timeout: 5 seconds wait for ready state

### Shutdown
- File: `orchestrator/main.py`
- Function: `run_orchestrator()` finally block
- Line: ~3365 (before capture.stop())
- Method: `mpd_manager.cleanup()` (safe no-op if not running)

## Error Handling

### MPD Not Found
```
✗ mpd command not found. Install MPD: apt install mpd (Debian/Ubuntu)
↓
Logger will continue, music features unavailable
```

### MPD Fails to Start
```
✗ MPD failed to start (exit code X). stderr: ...
↓
Logger warns, sets mpd_manager = None
↓
Music commands fail gracefully
```

### MPD Not Ready in Time
```
⚠ MPD started but did not become ready within timeout
↓
Logger warns, music may not be immediately available
↓
Orchestrator continues, may work after delay
```

## Next Steps

1. **Install MPD** (if you haven't already):
   ```bash
   sudo apt install mpd
   ```

2. **Test the integration**:
   ```bash
   python3 validate_mpd_integration.py
   ```

3. **Run the orchestrator**:
   ```bash
   python3 orchestrator/main.py
   ```

4. **Verify MPD is running**:
   ```bash
   ps aux | grep mpd
   mpc status
   ```

5. **Test music commands**:
   - In your voice assistant, try: "play some jazz"
   - Check logs for music skill execution
