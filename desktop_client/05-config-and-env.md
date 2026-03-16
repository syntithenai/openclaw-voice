# 05 — Config and .env Design

## Configuration layers

Precedence (highest to lowest):

1. Runtime settings saved by user (app config store)
2. Process environment variables
3. `.env` file defaults
4. Hardcoded safe defaults

## Required variables

- `DESKTOP_WEB_UI_URL`

## Optional variables

- `DESKTOP_WS_URL` (advanced override; auto-derived from `DESKTOP_WEB_UI_URL` when omitted)
- `DESKTOP_DEFAULT_TTS_MUTED` (default `false`)
- `DESKTOP_DEFAULT_CONTINUOUS_MODE` (default `false`)
- `DESKTOP_RECONNECT_DELAY_S` (default `1.5`)

## Proposed `.env` template

```dotenv
# Web UI link opened from tray menu
DESKTOP_WEB_UI_URL=http://127.0.0.1:18910

# App behavior
DESKTOP_DEFAULT_TTS_MUTED=false
DESKTOP_DEFAULT_CONTINUOUS_MODE=false
DESKTOP_RECONNECT_DELAY_S=1.5
```

## Persistence behavior

- First launch imports from `.env`.
- Subsequent updates write to app-local config storage.
- Optional “export current config to `.env.local`” action for backup.

## Validation rules

- `DESKTOP_WEB_UI_URL` must be a valid HTTP/HTTPS URL.
- `DESKTOP_RECONNECT_DELAY_S` must be a bounded positive number.
