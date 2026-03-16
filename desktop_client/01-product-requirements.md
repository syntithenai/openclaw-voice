# 01 — Product Requirements

## Objective

Build a desktop tray client that mirrors key control capabilities from the orchestrator web UI while staying lightweight, always available, and cross-platform.

## In-scope capabilities

1. **Tray microphone icon** visible in system tray/taskbar status area.
2. **Left click action** triggers the same behavior as the conference speaker play button in the web UI.
3. **Right click context menu** with:
   - Open Web UI
   - Mute TTS (toggle)
   - Continuous Mode (toggle)
   - Settings (opens modal)
4. **Settings modal fields**:
   - OpenClaw Voice orchestrator Web UI URL (`DESKTOP_WEB_UI_URL`)
5. **Microphone VU meter visualization** by border thickness, matching web UI semantics.
6. **Cross-platform support** for Linux, Windows, macOS.
7. **Environment-based bootstrap** using `.env` defaults for current setup.

## Non-goals (phase 1)

- Full chat UI replacement for the existing web app.
- Local audio capture pipeline reimplementation (desktop app consumes orchestrator telemetry only).
- Multi-user profile syncing.

## Functional requirements

- The client must start hidden to tray after first launch setup.
- Menu toggle state must reflect current orchestrator state on open.
- Settings changes must validate format and persist locally.
- Failure to reach orchestrator must not crash the app; display status and retry.

## Non-functional requirements

- Idle CPU usage target: low single-digit percent on typical desktop.
- Tray VU updates should remain smooth without excessive repaint churn.
- Cold start target: under 3 seconds on developer baseline hardware.
- No plaintext secret logging.

## Constraints and dependencies

- Behavior parity depends on stable orchestrator endpoint contracts.
- Linux tray behavior varies by desktop environment (GNOME/KDE/XFCE).
- macOS notarization/signing required for frictionless distribution.

## Success criteria

- User can trigger conference-speaker-equivalent action in one click.
- Menu toggles accurately control and reflect orchestrator state.
- VU meter tracks web UI level bands perceptibly and consistently.
- One installable artifact per OS family is available in CI releases.
