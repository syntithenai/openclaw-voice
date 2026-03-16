# 02 — Architecture

## Technology choice

Use **Tauri v2** as the default implementation.

### Rationale

- Strong cross-platform tray APIs.
- Small runtime footprint vs Electron.
- Rust side can securely handle secrets and OS integration.
- Frontend can reuse existing web patterns for VU/state display logic.

## High-level components

1. **Tray Host (Rust)**
   - App lifecycle
   - Tray icon rendering and menu
   - Native notifications and browser launch
2. **UI Window (Frontend)**
   - Settings modal content
   - Validation and save operations
3. **State Engine**
   - Poll/subscription to orchestrator state endpoints
   - State cache with freshness timestamps
4. **Command Gateway**
   - Executes left-click action and menu toggle operations
5. **Config Service**
   - Loads defaults from `.env`
   - Persists user overrides in app config storage

## Runtime data flow

1. App starts and loads env defaults + local overrides.
2. App health-checks orchestrator endpoint.
3. Tray icon initializes in disconnected/connected state.
4. Background task updates:
   - Toggle states
   - VU level
   - Connectivity status
5. User actions issue command calls and apply optimistic UI, then reconcile with server response.

## State model

- `connected: boolean`
- `mute_tts: boolean`
- `continuous_mode: boolean`
- `vu_level: float in [0,1]`
- `last_error: string | null`
- `updated_at: epoch_ms`

## Failure modes and handling

- **Network timeout**: mark stale state, keep UI responsive, retry with backoff.
- **401/403**: prompt to update keys in Settings.
- **5xx**: transient warning, no hard failure.
- **Invalid config**: prevent save and show field-level errors.

## Optional alternate architecture

- Electron + Node host for faster JS-only development.
- Keep API/state/config contracts identical so migration cost remains low.
