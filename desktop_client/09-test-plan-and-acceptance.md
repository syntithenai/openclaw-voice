# 09 — Test Plan and Acceptance

## Test layers

1. **Unit tests**
   - Config parsing/validation
   - VU smoothing and border mapping
   - Error classification and retry behavior
2. **Integration tests**
   - Command calls against test orchestrator
   - Toggle state reconciliation
3. **End-to-end smoke tests**
   - Tray launch
   - Left click command path
   - Right click menu toggles
   - Settings save and reconnect

## Acceptance criteria

- Left click reliably triggers conference-play behavior.
- Menu toggles modify backend state and remain synchronized.
- Settings persist across restart and mask secrets.
- VU border thickness visibly responds to audio level changes.
- App survives orchestrator downtime and reconnects cleanly.

## Cross-platform matrix

| Area | Linux | Windows | macOS |
|---|---|---|---|
| Tray icon render | ✅ | ✅ | ✅ |
| Left click action | ✅ | ✅ | ✅ |
| Right click menu | ✅ | ✅ | ✅ |
| Settings modal | ✅ | ✅ | ✅ |
| VU animation | ✅ | ✅ | ✅ |
| Startup + reconnect | ✅ | ✅ | ✅ |

## Manual exploratory checks

- Sleep/wake and network switch resilience.
- Invalid key handling and helpful recovery prompts.
- Mixed DPI/scale and multi-monitor tray behavior.
