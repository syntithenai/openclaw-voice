# Google Login How-To (Embedded Voice Web UI)

This guide configures Google Sign-In for the embedded web UI and secures both:

- HTTP UI access in required mode
- WebSocket access (`/ws`) using the same authenticated session cookie

## 1) Google Cloud Console setup

1. Open Google Cloud Console: https://console.cloud.google.com/
2. Select or create a project.
3. Configure OAuth consent screen:
   - Console page: https://console.cloud.google.com/apis/credentials/consent
   - Choose `Internal` (Workspace-only) or `External`.
   - Add app name and support email.
   - Add test users if app is not published and consent type is `External`.
4. Create OAuth client credentials:
   - Console page: https://console.cloud.google.com/apis/credentials
   - Create credentials -> OAuth client ID -> Web application.
5. Add Authorized redirect URIs for your UI endpoint:
   - `https://<your-host>/auth/google/callback`
   - For local testing, localhost callback URIs are allowed (for example `http://localhost:18910/auth/google/callback`).
6. Save, then download the client JSON file.

## 2) Reuse existing google_client_secret.json

This repo is configured to use the existing file by default:

- `WEB_UI_GOOGLE_CLIENT_SECRET_FILE=../google_client_secret.json`

Given this workspace layout, that resolves to:

- `/home/stever/projects/openclawstuff/google_client_secret.json`

No duplication is required.

## 3) Configure environment

In your active profile (`.env`, `.env.docker`, or `.env.pi`):

- `WEB_UI_ENABLED=true`
- `WEB_UI_AUTH_MODE=required` (or `optional`)
- `WEB_UI_GOOGLE_CLIENT_SECRET_FILE=../google_client_secret.json`
- `WEB_UI_AUTH_COOKIE_SECURE=true`
- `WEB_UI_SSL_CERTFILE=<your cert pem>`
- `WEB_UI_SSL_KEYFILE=<your key pem>`

Optional settings:

- `WEB_UI_GOOGLE_REDIRECT_URI=https://<your-host>/auth/google/callback`
  - Leave empty to auto-derive from request host/scheme.
- `WEB_UI_GOOGLE_ALLOWED_DOMAIN=example.com`
  - Restricts login to a single email domain.
- `WEB_UI_AUTH_SESSION_TTL_HOURS=24`
- `WEB_UI_AUTH_SESSION_COOKIE_NAME=openclaw_ui_session`

If you run on plain HTTP in non-localhost environments, Google OAuth may fail due to redirect URI policy. Prefer HTTPS.

## 4) Runtime behavior

- `WEB_UI_AUTH_MODE=disabled`
  - No login required.
- `WEB_UI_AUTH_MODE=optional`
  - Login button shown in header, app remains usable without login.
- `WEB_UI_AUTH_MODE=required`
  - If not logged in, the main page renders a login-required warning only.
  - WebSocket connections are rejected until authenticated.
  - Protected file routes are denied when unauthenticated.

## 5) Endpoints

- Session status: `GET /auth/session`
- Start login: `GET /auth/google/login`
- OAuth callback: `GET /auth/google/callback`
- Logout: `POST /auth/logout` (or `GET /auth/logout`)

The browser receives an HttpOnly session cookie after successful callback, and that same cookie is used by WebSocket authentication.

## 6) Quick verification

1. Open UI while logged out with `WEB_UI_AUTH_MODE=required`.
2. Confirm main content shows login warning only.
3. Click Sign in and complete Google flow.
4. Confirm full UI appears and realtime status updates work.
5. Log out and confirm UI returns to warning state and WS is blocked.
