# 11 — Implementation Guide

This document maps the design sequence to the implemented desktop client code in this folder.

## Implemented components

- `client/config.py` — Loads desktop settings from `desktop_client/.env` with fallback to project `.env`; derives WS URL from `DESKTOP_WEB_UI_URL`.
- `client/realtime.py` — WebSocket bridge to orchestrator (`/ws`) with auto-reconnect.
- `client/controller.py` — UI behavior contract:
  - left click / default action -> `{"type":"mic_toggle"}`
  - mute toggle -> `{"type":"tts_mute_set","enabled":...}`
  - continuous toggle -> `{"type":"continuous_mode_set","enabled":...}`
- `client/vu.py` — VU border thickness mapping matching web UI.
- `client/tray.py` — System tray icon + right-click menu + settings launch.
- `client/settings_ui.py` — Settings modal and validation (Web UI URL only).
- `tests/test_ui_behavior.py` — Automated UI behavior tests.

## Run locally

1. Install dependencies from `desktop_client/requirements.txt`.
2. Start orchestrator with web UI enabled (`WEB_UI_PORT`, `WEB_UI_WS_PORT`).
3. Run the desktop client module: `python -m client` from `desktop_client/`.

## Automated UI testing

Run `pytest desktop_client/tests -q` from the repository root.

The tests validate:

- Left-click action mapping parity (`mic_toggle`)
- Right-click toggle payloads (`tts_mute_set`, `continuous_mode_set`)
- Incoming state snapshot synchronization
- VU border mapping parity to web UI formula
- Settings modal validation contract

## Packaging notes

For cross-platform builds, package with PyInstaller in CI targets:

- Linux: one-folder build + desktop entry
- Windows: exe + installer (MSI/NSIS)
- macOS: app bundle + notarization step
