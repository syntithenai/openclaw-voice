# MPD Lifecycle Management

## Overview

As of this update, **MPD (Music Player Daemon) is now managed by the orchestrator** rather than docker-compose. The orchestrator:
- Starts MPD during its initialization
- Stops MPD during graceful shutdown
- Monitors MPD health and readiness

## Architecture

Current runtime model:
```
orchestrator (host or container)
└── starts/manages local mpd process
```

## Installation Requirements

MPD must be installed on the system where the orchestrator runs.

### On Linux (Debian/Ubuntu):
```bash
sudo apt install mpd
```

### Verify Installation:
```bash
which mpd
mpd --version
```

## Configuration

### MPD Config File Locations

The orchestrator searches for `mpd.conf` in this order:
1. `OPENCLAW_MPD_CONFIG` (explicit override)
2. `~/.config/mpd/mpd.conf` (user config)
3. `orchestrator/services/mpd.conf` (bundled container-safe config)
4. `/etc/mpd.conf` (system config)
5. Default MPD configuration if none found

### Example ~/.config/mpd/mpd.conf

```ini
music_directory "~/Music"
playlist_directory "~/.config/mpd/playlists"
db_file "~/.config/mpd/database"
log_file "~/.config/mpd/mpd.log"
pid_file "~/.config/mpd/mpd.pid"
state_file "~/.config/mpd/state"

bind_to_address "127.0.0.1"
port "6600"

audio_output {
    type "pipewire"
    name "PipeWire Output"
}

audio_output {
    type "alsa"
    name "ALSA Output"
}

log_level "default"
auto_update "yes"
restore_paused "yes"
```

## Running the Orchestrator

### Host-based Orchestrator (Recommended)

```bash
cd /home/stever/projects/openclawstuff/openclaw-voice
python3 orchestrator/main.py
```

Expected startup sequence:
```
→ OpenClaw Voice Orchestrator - Initializing
→ Validating runtime configuration...
→ Loading VAD model...
✓ VAD loaded in XXXms
→ Initializing gateway...
✓ Gateway ready in XXXms
→ Starting MPD...
✓ MPD ready in XXXms
```

### Docker-based Orchestrator

If running orchestrator in a container, MPD is already bundled inside that image.

```bash
# Build and start the orchestrator service normally.
# The container will start MPD locally during orchestrator init.
```

## Troubleshooting

### "mpd command not found"

MPD is not installed or not in PATH:
```bash
# Install on Linux
sudo apt install mpd

# Verify
which mpd
```

### "MPD failed to start (exit code ...)"

Check the error message:
- **Port already in use**: Another MPD instance or service is using port 6600
  ```bash
  sudo lsof -i :6600
  kill <PID>
  ```

- **Permission denied**: Usually related to music directory or PID file location
  ```bash
  mkdir -p ~/.config/mpd ~/.mpdstate
  chmod 700 ~/.config/mpd ~/.mpdstate
  ```

- **Missing config**: Create a basic RPD config in `~/.config/mpd/mpd.conf`

### "MPD did not become ready within 10s"

- Verify MPD process is running: `ps aux | grep mpd`
- Check MPD logs: `~/.config/mpd/mpd.log`
- Test connection: `nc -zv 127.0.0.1 6600`
- Restart orchestrator

### Music Features Unavailable

If orchestrator logs "Failed to start MPD; music features may be unavailable":
- Music commands will fail gracefully
- Verify MPD is installed and can be started manually:
  ```bash
  mpd ~/.config/mpd/mpd.conf
  mpc play
  mpc stop
  ```

## Integration Points

MPD Lifecycle is managed in:
- **File**: `orchestrator/services/mpd_manager.py`
- **Integration**: `orchestrator/main.py` lines ~960 (startup) and ~3370 (shutdown)

## Supported Docker Model

Docker deployments use the orchestrator-managed MPD runtime.

## Related Files

- `orchestrator/services/mpd_manager.py` - MPD lifecycle implementation
- `orchestrator/services/mpd.conf` - Bundled container-safe MPD configuration
- `orchestrator/main.py` - Integration point (~line 960 for startup, ~3370 for shutdown)
- `docker-compose.yml` - Orchestrator services now point MPD traffic to `127.0.0.1`
- `.env` / `.env.docker` - MPD connection settings
