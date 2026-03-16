# 04 — API Contracts

## Goal

Define the client-to-orchestrator API surface required for tray behavior parity.

## Contract strategy

- Reuse existing web UI endpoints where available.
- If endpoint names differ, preserve intent and map in one adapter layer.
- Keep desktop client insulated from backend renames via local command aliases.

## Required operations

1. **Trigger conference speaker play behavior**
   - Alias: `trigger_conference_play`
   - Method: POST
   - Auth: key/header from config

2. **Read and toggle Mute TTS**
   - `get_mute_tts_state` (GET)
   - `set_mute_tts_state` (POST/PATCH)

3. **Read and toggle Continuous Mode**
   - `get_continuous_mode_state` (GET)
   - `set_continuous_mode_state` (POST/PATCH)

4. **Read VU level**
   - `get_vu_level` (GET or WebSocket stream)
   - Value normalized to $[0,1]$.

5. **Health and version**
   - `get_orchestrator_health`
   - `get_orchestrator_version`

## Example desktop-side command interface

- `commands/play`
- `commands/toggles/mute-tts`
- `commands/toggles/continuous-mode`
- `telemetry/vu`
- `system/health`

These are local adapter names, not mandatory backend paths.

## Error contract

Map backend failures to user-facing classes:

- `AUTH_ERROR`
- `NETWORK_ERROR`
- `VALIDATION_ERROR`
- `SERVER_ERROR`
- `UNKNOWN_ERROR`

## Sync model

- On menu open: force refresh toggle state.
- Background sync cadence: configurable interval.
- After command: optimistic local update, then hard reconcile.
