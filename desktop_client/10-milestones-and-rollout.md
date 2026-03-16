# 10 — Milestones and Rollout

## Milestone 0 — Contract lock (1–2 days)

- Verify and document exact orchestrator endpoints used by web UI actions.
- Freeze desktop adapter command names.
- Exit criteria: API contract approved.

## Milestone 1 — Core tray prototype (3–5 days)

- Tray icon, left-click behavior, right-click menu shell.
- Basic connectivity indicator.
- Exit criteria: click path works end-to-end.

## Milestone 2 — Settings and config (2–4 days)

- Settings modal, field validation, persistence.
- `.env` bootstrap and override handling.
- Exit criteria: config survives restarts.

## Milestone 3 — VU meter parity (2–3 days)

- VU ingestion, smoothing, thickness mapping.
- Visual parity calibration vs web UI.
- Exit criteria: parity sign-off in side-by-side test.

## Milestone 4 — Cross-platform hardening (4–7 days)

- Linux/Windows/macOS QA passes.
- Packaging artifacts generated in CI.
- Exit criteria: install and run validated on all target OSes.

## Milestone 5 — Release candidate and docs (2–3 days)

- Publish install docs and troubleshooting.
- Cut RC build and gather feedback.
- Exit criteria: no blocker defects for GA.

## Post-GA enhancements

- Auto-update channel.
- Optional hotkey trigger for play action.
- Rich status diagnostics panel.
