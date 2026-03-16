# 03 — Tray UX and Interactions

## Interaction model

## Left click

- Action: invoke the same backend operation used by the web conference speaker play button.
- Visual feedback:
  - Immediate pulse animation on tray icon.
  - Success: brief accent highlight.
  - Failure: warning notification with short reason.

## Right click menu

Menu order:

1. **Open Web UI**
2. Separator
3. **Mute TTS** (checkmark toggle)
4. **Continuous Mode** (checkmark toggle)
5. Separator
6. **Settings…**
7. Separator
8. **Reconnect** (manual refresh)
9. **Quit**

## Status conventions

- Connected: normal mic icon + VU border animation.
- Degraded: dim icon + tooltip “Connection unstable”.
- Disconnected: slashed mic icon + tooltip “Disconnected”.

## Settings modal UX

Sections:

1. **Orchestrator Connection**
   - Web UI URL (`DESKTOP_WEB_UI_URL`)

Actions:

- `Save`
- `Cancel`

Validation:

- Web UI URL must be absolute HTTP/HTTPS.
- Save disabled until form is valid.

## Accessibility

- Keyboard navigable modal and controls.
- High contrast icon variants.
- Descriptive labels and error text.
