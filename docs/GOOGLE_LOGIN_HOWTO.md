# Google Login How-To (Embedded Voice Web UI)

This guide configures **client-side Google Sign-In** for the embedded web UI—**no internet-exposed callback endpoints required**.

Authentication secures:
- HTTP UI access in required mode
- WebSocket access (`/ws`) using the same authenticated session cookie

## Architecture: Client-Side OAuth (No Exposed Callbacks)

1. Browser loads Google Sign-In JS library
2. User clicks "Sign in" → Google auth happens entirely in the browser
3. Browser sends ID token to local `POST /auth/google/token` endpoint
4. Server verifies token, creates session, sets HttpOnly cookie
5. WebSocket connections check this session cookie

**Key benefit:** Google redirects stay within the browser; server endpoint doesn't need internet access.

## 1) Google Cloud Console Setup

1. Open Google Cloud Console: https://console.cloud.google.com/
2. Select or create a project.
3. Configure OAuth consent screen:
   - Console page: https://console.cloud.google.com/apis/credentials/consent
   - Choose `Internal` (Workspace-only) or `External`.
   - Add app name and support email.
   - Add test users if app is not published and consent type is `External`.
4. Create OAuth client credentials:
   - Console page: https://console.cloud.google.com/apis/credentials
   - Create credentials → OAuth client ID → **Web application** (not Desktop/Mobile).
5. **Do NOT add Redirect URIs** in the credentials (client-side flow uses the browser).
6. Save, then download the client JSON file.

## 2) Reuse or Create google_client_secret.json

Download the client JSON from Google Cloud Console and save it as:

- `/home/stever/projects/openclawstuff/google_client_secret.json` (parent of openclawstuff directory)

This repo is configured to use that location by default:
- `WEB_UI_GOOGLE_CLIENT_SECRET_FILE=../google_client_secret.json`

If the file doesn't exist, you can also set credentials inline via environment:
- `WEB_UI_GOOGLE_CLIENT_ID=<your-client-id>`
- `WEB_UI_GOOGLE_CLIENT_SECRET=<your-client-secret>`

## 3) Configure Environment

In your active profile (`.env`, `.env.docker`, or `.env.pi`):

**Required:**
- `WEB_UI_ENABLED=true`
- `WEB_UI_AUTH_MODE=required` (or `optional`)
- `WEB_UI_GOOGLE_CLIENT_SECRET_FILE=../google_client_secret.json` (or set inline credentials)
- `WEB_UI_AUTH_COOKIE_SECURE=true`

**For remote access (recommended):**
- `WEB_UI_SSL_CERTFILE=<your cert pem>`
- `WEB_UI_SSL_KEYFILE=<your key pem>`

**Optional:**
- `WEB_UI_GOOGLE_ALLOWED_DOMAIN=example.com`
  - Restricts login to a single email domain (e.g., your organization).
- `WEB_UI_AUTH_SESSION_TTL_HOURS=24`
- `WEB_UI_AUTH_SESSION_COOKIE_NAME=openclaw_ui_session`

For **localhost-only access**, HTTPS is not strictly required. For **remote access**, HTTPS is recommended.

## 4) How It Works at Runtime

### Auth Modes

- `WEB_UI_AUTH_MODE=disabled`
  - No login required.
- `WEB_UI_AUTH_MODE=optional`
  - Login button shown in header; app remains usable without login.
- `WEB_UI_AUTH_MODE=required`
  - If not logged in, main page renders a login-required warning.
  - WebSocket connections are rejected (close code 4401) until authenticated.
  - Protected file routes (`/files/workspace`, `/files/media`, `/recordings/audio/`) denied when unauthenticated.

### Login Flow

1. User clicks "Sign in" button.
2. Modal appears with Google Sign-In button.
3. Google auth prompt opens in browser (no server redirect).
4. After user consents, Google returns an ID token to the browser JS.
5. Browser sends token to `POST /auth/google/token`.
6. Server verifies token via Google's API (`tokeninfo` endpoint).
7. Server creates session and returns `Set-Cookie` header.
8. Full UI appears; WebSocket connects using the session cookie.

### Logout Flow

1. User clicks logout button.
2. `POST /auth/logout` clears session.
3. Browser is revoked from Google (optional).
4. Page returns to login-required warning state.

## 5) Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/auth/session` | Get current auth state (mode, authenticated, user) |
| POST | `/auth/google/token` | **Client submits Google ID token here** (no internet exposure needed) |
| POST | `/auth/logout` | Clear session and logout |

## 6) Security Properties

- **No callback endpoint exposed:** Google redirects happen in the browser, not server.
- **Token verification:** Server validates ID token via Google's public API.
- **Session cookie:** HttpOnly, Secure (HTTPS), SameSite=Lax.
- **WebSocket gate:** Session cookie required for `/ws` in required mode.
- **Domain restriction:** Optional email domain allowlist prevents unauthorized Google accounts.

## 7) Quick Verification

1. Start orchestrator with `WEB_UI_AUTH_MODE=required` in your .env.
2. Open UI while logged out.
3. Confirm main content shows login warning only.
4. Click "Sign in" button.
5. A modal appears with the Google Sign-In button.
6. Complete Google OAuth flow (consent screen).
7. Confirm full UI appears and realtime status updates work.
8. Click logout and confirm page returns to warning state and WS closes.

## 8) Troubleshooting

**"Google Sign-In button not appearing"**
- Ensure `WEB_UI_GOOGLE_CLIENT_SECRET_FILE` points to a valid JSON file, or `WEB_UI_GOOGLE_CLIENT_ID` is set.
- Check browser console for JS errors.
- Confirm `WEB_UI_AUTH_MODE` is not `disabled`.

**"Token verification failed"**
- Verify Google client ID matches what's in the credentials JSON.
- Ensure the email being used to sign in belongs to an allowed domain (if `WEB_UI_GOOGLE_ALLOWED_DOMAIN` is set).
- Check server logs for detailed error messages.

**"HTTPS required but I'm on localhost"**
- Localhost HTTP is allowed by Google. For production, use HTTPS.
- Set `WEB_UI_SSL_CERTFILE` and `WEB_UI_SSL_KEYFILE` to enable HTTPS.

**"Session expires too quickly"**
- Adjust `WEB_UI_AUTH_SESSION_TTL_HOURS` (default: 24 hours).
- Must be > 0.
