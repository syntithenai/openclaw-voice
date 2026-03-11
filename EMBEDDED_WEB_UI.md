# Embedded Realtime Voice UI

This orchestrator can expose a lightweight web UI + WebSocket bridge for continuous browser audio mode.

## What it provides

- Browser microphone capture with continuous streaming to the orchestrator over WebSocket
- Browser-side VU meter (local mic level)
- Realtime orchestrator status indicators:
  - sleep / awake
  - speech activity
  - hotword activity
  - TTS speaking state
- Orchestrator microphone VU meter
- Embeddable UI (e.g. iframe in OpenClaw web UI)

## Enable it

Set in your env profile (`.env`, `.env.docker`, or `.env.pi`):

- `WEB_UI_ENABLED=true`
- `WEB_UI_HOST=0.0.0.0`
- `WEB_UI_PORT=18910`
- `WEB_UI_WS_PORT=18911`
- `WEB_UI_STATUS_HZ=12`
- `WEB_UI_HOTWORD_ACTIVE_MS=2000`

Then restart the native orchestrator.

## URLs

- UI: `http://<host>:<WEB_UI_PORT>/`
- Health: `http://<host>:<WEB_UI_PORT>/health`
- WebSocket: `ws://<host>:<WEB_UI_WS_PORT>/ws`

## Embed example

Use in a third-party page:

`<iframe src="http://VOICE_HOST:18910/" style="width:100%;height:420px;border:0"></iframe>`

## WebSocket payloads

From browser UI to orchestrator:

- Text frame (level telemetry):
  - `{"type":"browser_audio_level","rms":0.031,"peak":0.22}`
- Binary frame (optional):
  - little-endian `int16` PCM chunk

From orchestrator to UI clients:

- `{"type":"status", "orchestrator": {...}, "browser_audio": {...}, "connections": {...}}`

`orchestrator` includes:

- `voice_state`
- `wake_state`
- `speech_active`
- `hotword_active`
- `tts_playing`
- `mic_rms`
- `queue_depth`
