# Desktop Client Design Pack

This folder contains the detailed design sequence for a cross-platform tray desktop client for the `openclaw-voice` orchestrator.

## Document sequence

1. `01-product-requirements.md` — scope, goals, and non-goals
2. `02-architecture.md` — technical architecture and runtime model
3. `03-tray-ux-and-interactions.md` — left/right click behavior, menu, and settings modal UX
4. `04-api-contracts.md` — API mapping to orchestrator behavior
5. `05-config-and-env.md` — `.env` strategy and settings persistence
6. `06-vu-meter-design.md` — tray microphone VU meter model
7. `07-security-and-secrets.md` — key handling and threat boundaries
8. `08-cross-platform-build-and-packaging.md` — Linux/Windows/macOS delivery
9. `09-test-plan-and-acceptance.md` — test matrix and release criteria
10. `10-milestones-and-rollout.md` — phased implementation timeline
11. `11-implementation-guide.md` — code mapping, run/test instructions, packaging notes

## Primary decision

The baseline architecture uses **Tauri v2** with:

- Rust host for tray integration and secure local storage
- Small web frontend for settings modal/state rendering
- HTTP/WebSocket client layer to orchestrator web endpoints

Alternative implementation notes are included where relevant.
