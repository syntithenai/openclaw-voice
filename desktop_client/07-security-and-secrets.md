# 07 — Security and Secrets

## Security goals

- Keep API keys out of source control and logs.
- Minimize exposure in process memory and UI surfaces.
- Prevent accidental leak via diagnostics.

## Secret handling policy

1. Load secrets from `.env` only as bootstrap defaults.
2. Store overrides in OS-appropriate secure storage when available:
   - macOS Keychain
   - Windows Credential Manager
   - Linux Secret Service/libsecret
3. If secure storage unavailable, encrypt at rest using app-generated local key with user warning.

## Logging policy

- Never log full keys.
- Masked form only: first 3 and last 2 chars.
- Redact headers and query params containing token-like keys.

## Network security

- Prefer HTTPS endpoints.
- Warn when using HTTP non-localhost targets.
- Set strict request timeout and retry caps.

## Threat boundaries

- Desktop client is a convenience control surface, not identity provider.
- Trust anchored in orchestrator auth and local machine trust.
- Compromised local user session implies elevated risk; document this explicitly.

## Compliance checklist (phase 1)

- [ ] `.env` excluded from git
- [ ] secret masking unit tests
- [ ] no secret values in crash dumps/telemetry
- [ ] transport warnings for insecure endpoints
