# Playlist Load Latency Investigation and Remediation Plan

Date: 2026-03-21
Scope: OpenClaw Voice orchestrator + embedded Web UI + MPD integration
Status: Planning only (no runtime behavior changes in this step)

## 1) Problem Statement

Observed user-facing latency for `music_load_playlist` can exceed 20s, while direct `mpc` usage is fast.

Key objective:
- Identify exactly where latency is accumulated between user action and visible UI update.
- Prove root cause with automated measurements.
- Choose the highest-impact fix based on data, not tuning intuition.

## 2) Primary Hypotheses (to prove/disprove)

H1. **MPD is not the bottleneck** for `clear/load/status`; orchestrator scheduling/queueing adds most delay.

H2. **Head-of-line blocking in shared MPD pool** causes control commands (`clear`, `load`) to wait behind long list queries (`playlistinfo`).

H3. **UI ACK/state contract sequencing** introduces user-visible wait because ACK is delayed until slow handler stages complete.

H4. **Publisher-induced contention** (queue sync + playlist updates + snapshot pushes) amplifies load latency under concurrency.

## 3) Latency Model and Required Metrics

Define end-to-end latency as:

```
T_total = T_ws_in
        + T_handler_queue_wait
        + T_mpd_conn_acquire_wait
        + T_mpd_clear
        + T_mpd_load
        + T_ack_emit
        + T_state_push_sched
        + T_state_push_exec
        + T_browser_render
```

### Mandatory measured timestamps per action_id

1. Browser UI
- `ui_click_ts`
- `ui_ws_send_ts`
- `ui_ack_recv_ts`
- `ui_transport_update_ts`
- `ui_queue_update_ts`
- `ui_render_settled_ts`

2. Realtime service (WebSocket server)
- `ws_msg_received_ts`
- `music_load_handler_enter_ts`
- `music_load_handler_exit_ts`
- `ack_send_start_ts`
- `ack_send_done_ts`
- `state_push_scheduled_ts`
- `state_push_done_ts`

3. Music manager / MPD pool
- `load_playlist_enter_ts`
- `list_playlists_start_ts`, `list_playlists_end_ts`
- `mpd_clear_call_ts`, `mpd_clear_done_ts`
- `mpd_load_call_ts`, `mpd_load_done_ts`
- `load_playlist_return_ts`

4. MPD pool internals (critical)
- `pool_get_request_ts`
- `pool_get_grant_ts`
- `pool_wait_ms`
- `conn_id`
- `command_kind` (`control`, `list`, `search`, `status`)

### Metric outputs

- Per-stage latency distributions: p50/p95/p99/max
- Stage contribution percentages to `T_total`
- Timeout and retry rates by command type
- Correlation between `pool_wait_ms` and end-to-end delay

## 4) Baseline Comparisons to Automate

All baselines run against the same playlists (50, 200, 500 tracks), same machine, same MPD DB.

### A) Raw MPD CLI baseline

Measure command latency with `mpc`:
- `mpc clear`
- `mpc load <playlist>`
- `mpc status`

Collect N=50 runs per size, output CSV.

### B) Raw MPD protocol baseline (optional but recommended)

Direct TCP client to MPD (no orchestrator) issuing equivalent commands to validate CLI overhead is negligible.

### C) Orchestrator internal baseline

Trigger `music_load_playlist` through WS API from script client.
Record server-side stage timings and ACK latency.

### D) Browser E2E baseline

Use Playwright/Selenium flow:
- Click playlist load in real UI
- Capture ACK + visible transport + queue paint timings

## 5) Automation Harness Design

## Proposed files (to be implemented)

- `benchmarks/playlist-load/run_playlist_load_bench.py`
  - Orchestrates all scenarios
  - Generates run IDs and writes artifacts

- `benchmarks/playlist-load/mpc_baseline.py`
  - Runs raw `mpc` timing loops

- `benchmarks/playlist-load/ws_action_client.py`
  - Sends `music_load_playlist` with `action_id`
  - Waits for ACK + state updates

- `test/selenium_playlist_load_latency.py` (or Playwright equivalent)
  - Captures browser-side timings

- `benchmarks/playlist-load/analyze_playlist_load.py`
  - Produces summary markdown + CSV + plots

### Artifact structure

`benchmarks/playlist-load/results/<run_id>/`
- `env.json` (machine, mpd version, config snapshot)
- `mpc_baseline.csv`
- `ws_runs.csv`
- `browser_runs.csv`
- `server_traces.jsonl`
- `summary.md`
- `waterfalls/*.png`

## 6) Scenario Matrix

Minimum scenarios:

1. **Idle baseline**
- No concurrent UI sync actions

2. **Queue-sync pressure**
- Induce periodic queue reads while loading playlist

3. **Multi-client contention**
- 1, 2, 4 WebSocket clients issuing load/snapshot actions

4. **Large queue + frequent state polling**
- Stress current synchronization pathway

5. **Network impairment (optional)**
- Add loopback latency/jitter to isolate transport effects

For each scenario:
- Playlist sizes: 50 / 200 / 500
- Repetitions: 30–50
- Report p50/p95/p99

## 7) SLOs and Pass/Fail Gates

Initial target gates:
- ACK latency p95 <= 1.5s
- Transport update latency p95 <= 2.0s
- Queue visible update p95 <= 4.0s
- Timeout rate for load path < 0.5%

Any failed gate must include stage-level attribution (which term dominates `T_total`).

## 8) Fix Decision Tree (data-driven)

If `pool_wait_ms` dominates:
- **Fix A (high priority):** split MPD pools by QoS
  - Control pool: `clear/load/play/pause/stop`
  - Data pool: `playlistinfo/search/listplaylists`

If ACK waits on non-critical work:
- **Fix B:** send ACK immediately after control-plane success; move sync work fully async.

If queue refresh causes repeated long commands:
- **Fix C:** event-driven queue invalidation (MPD `idle`/subsystem changes), no periodic heavy polling.

If UI rendering dominates:
- **Fix D:** virtualized list rendering + incremental queue hydration.

If MPD itself is slow only at large sizes:
- **Fix E:** cached queue snapshot service with bounded incremental refresh.

## 9) Outside-the-Box Alternatives

1. **Dual-path architecture**
- Control path with strict low-latency budget and separate worker.
- Data path eventually consistent with stale-while-revalidate semantics.

2. **Queue mirror store**
- Maintain authoritative local queue mirror fed by MPD change events.
- UI reads mirror, not MPD directly, for most interactions.

3. **Action timeline debugger in UI**
- Expose per-action waterfall to developers/users for immediate diagnosis.

4. **Adaptive sync mode**
- Automatically degrade queue-detail refresh frequency when command latency rises.

## 10) Execution Phases

Phase 1 (1 day): instrumentation + raw MPD and orchestrator baselines

Phase 2 (1 day): browser E2E timing capture + full scenario matrix

Phase 3 (0.5 day): root-cause report with ranked fix candidates

Phase 4 (1–2 days): implement top fix behind feature flag

Phase 5 (0.5 day): rerun benchmarks, compare before/after, decide rollout

## 11) Definition of Done

- Reproducible benchmark command produces full artifacts with one command.
- Root-cause report identifies dominant stage(s) with p95/p99 evidence.
- At least one fix validated against same benchmark suite.
- User-visible playlist load latency improved and meets agreed SLOs.
