# MPD FIFO Audio Architecture (Draft)

This document defines the FIFO-based design where only the orchestrator owns playback hardware while MPD remains the music library/queue/control engine.

## Goals

- Eliminate direct hardware contention between MPD and orchestrator.
- Keep MPD strengths (library index, queueing, genre/artist operations).
- Centralize playback ownership in orchestrator (music + TTS + alarms).
- Preserve existing MPD control protocol integration.

## Target Signal Flow

1. Voice request reaches orchestrator music router.
2. Orchestrator sends MPD control commands (queue/play/stop/etc.).
3. MPD renders PCM to FIFO output (`type "fifo"`).
4. Orchestrator reads FIFO PCM via `MPDFifoReader`.
5. Future mixer stage combines FIFO music + TTS + alarms.
6. Orchestrator playback backend writes the mixed stream to hardware.

## Runtime Ownership Model

- **MPD owns**: music DB, queue, search, playback state machine.
- **Orchestrator owns**: physical output device access.
- **Shared boundary**: FIFO PCM file path.

## Deployment Matrix

- Docker orchestrator container (bundled MPD)
  - Host bind path: `MPD_FIFO_HOST_PATH` (default `/tmp/openclaw-mpd-fifo`)
  - Container FIFO path: `/tmp/mpd-fifo/music.pcm`
- Native orchestrator + native MPD
  - Both use same host path directly

## Step 1 (implemented now)

- MPD FIFO output is configured.
- Shared FIFO path is mounted into MPD and orchestrator containers.
- Orchestrator has a FIFO reader scaffold with lifecycle and ingest stats.
- No mixer integration yet; FIFO data is read and cached only.

## Step 2 (implemented now)

- FIFO PCM is played through orchestrator-owned playback in a background passthrough loop.
- Supports 16-bit PCM, mono/stereo input (stereo is downmixed to mono).
- Resamples FIFO stream to orchestrator playback sample rate when needed.
- Gating: pauses passthrough while TTS/alarm/listening paths are active.

## Next Phases

### Phase 2: Mixer Integration

- Add continuous playback mixer in orchestrator.
- Pull chunks from `MPDFifoReader` and mix with TTS/alarm streams.
- Handle sample rate/channel conversion where needed.

### Phase 3: Playback Routing Hardening

- Disable direct MPD ALSA/Pulse outputs for normal mode.
- Keep a feature flag for fallback direct MPD output when troubleshooting.

### Phase 4: Observability

- Expose FIFO ingest stats in logs/web UI.
- Add underrun/overrun counters and latency telemetry.

## Files Changed In Step 1

- `orchestrator/services/mpd.conf`
  - Added FIFO `audio_output` stanza.
- `docker-compose.yml`
  - Added shared FIFO bind path mounted at `/tmp/mpd-fifo` for orchestrator and optional `snapserver`.
- `orchestrator/config.py`
  - Added `MPD_FIFO_*` settings and validation.
- `orchestrator/audio/mpd_fifo_reader.py`
  - New scaffold reader module.
- `orchestrator/main.py`
  - Added startup/shutdown lifecycle for FIFO reader.
- `.env.example`, `.env.docker.example`, `README.md`
  - Added FIFO configuration docs/examples.

## Planned Files For Phase 2

- `orchestrator/audio/playback.py`
  - Add mixed stream ingestion path.
- `orchestrator/audio/duplex.py`
  - Add mixed output support for duplex mode.
- `orchestrator/main.py`
  - Wire mixer task and source priorities (music/TTS/alarm).
- `orchestrator/audio/resample.py`
  - Reuse/extend for music stream conversion if MPD FIFO format differs.

## Configuration Knobs

- `MPD_FIFO_ENABLED`
- `MPD_FIFO_PATH`
- `MPD_FIFO_SAMPLE_RATE`
- `MPD_FIFO_CHANNELS`
- `MPD_FIFO_BITS_PER_SAMPLE`

## Known Limitations (current step)

- Reader is not yet connected to audible output.
- Reader currently caches chunks for diagnostics/future consumption only.
- MPD output routing still depends on valid ALSA/Pulse access inside the orchestrator runtime.

## Network Sync Option (Suggestion Stage)

If synchronized multi-room playback is required, Snapcast is the best fit at this stage:

1. Keep MPD as source and expose PCM stream (FIFO or pipe command).
2. Feed Snapserver from MPD source stream.
3. Run Snapclient(s) on target devices for synchronized playback.
4. Keep orchestrator as control plane (play/pause/volume commands) while Snapcast handles transport/sync.

Suggested incremental path:

- First: make local FIFO mixer path audible (Phase 2).
- Next: add optional parallel MPD->Snapserver source mode behind a config flag (Docker profile `snapcast` now scaffolded).
- Finally: expose a simple "play to synchronized group" command in music router.
