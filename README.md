# OpenClaw Voice Orchestrator (Python)

This is the Python rebuild of the OpenClaw voice orchestrator. It runs **outside Docker** for fast iteration and direct audio device access. Docker is only used for optional Whisper/Piper services.

## Goals
- Cross‑platform: Linux, Raspberry Pi, Windows (best‑effort)
- Minimal dependencies
- Continuous audio capture with pre‑roll
- Wake word integration (OpenWakeWord)
- Silero VAD with optional WebRTC VAD override
- WebRTC AEC only
- Emotion tagging via SenseVoice (small model)

## Quick start (local orchestrator)
1. Create and activate a Python venv.
2. Install dependencies from `requirements.txt`.
3. Copy `.env` and edit device/config values.
4. Run `python -m orchestrator.main`.

### Optional dependencies
For wake word engines and SenseVoice emotion tagging, install optional packages from `requirements-optional.txt`.

### Tools
**Wake word test server**
- Run: `python -m orchestrator.tools.wakeword_test_server`
- POST a WAV file to `http://localhost:18950/test/wakeword`

**Wake word live test server (no upload)**
- Run: `python -m orchestrator.tools.wakeword_live_test`
- POST to `http://localhost:18952/test/wakeword-live`
- Optional header: `X-Duration-Seconds: 2.5`

**AEC test server**
- Run: `python -m orchestrator.tools.aec_test_server`
- POST JSON to `http://localhost:18951/test/aec` with base64 fields: `mic_wav`, `playback_wav`
- Or POST multipart form-data with files: `mic` and `playback`
- Response includes `rms_before`, `rms_after`, and `reduction_ratio`

**Silero model downloader**
- Run: `python -m orchestrator.tools.download_silero`

**Wake word CLI test (no server)**
- Run: `python -m orchestrator.tools.wakeword_cli_test --duration 2.0`

### WebRTC AEC bindings
Install `webrtc-audio-processing` to enable the WebRTC AEC adapter.

## Docker services
Sample Whisper and Piper services live under `docker/`. The orchestrator can call them but does not require Docker to run.

- `docker/whisper` — FastAPI + faster-whisper
- `docker/piper` — FastAPI + Piper
- `docker/whisper-models` — model selection file (required by the docker setup)

## Notes
This scaffold provides module stubs and a starting point. Implementation will follow the plan in `VOICE_ORCHESTRATOR_PYTHON_PLAN.md`.
