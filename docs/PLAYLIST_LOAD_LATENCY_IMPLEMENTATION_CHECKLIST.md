# Playlist Load Latency — Implementation Checklist

Date: 2026-03-21
Depends on: `PLAYLIST_LOAD_LATENCY_INVESTIGATION_PLAN.md`

## Goal

Implement an automated benchmark that pinpoints where playlist-load latency is spent, then use that data to choose and validate the best fix.

## Phase 0 — Guardrails

- [ ] Add feature flag: `OPENCLAW_MUSIC_LATENCY_TRACE=1` (default off)
- [ ] Ensure tracing emits one JSON line per event (low overhead, append-only)
- [ ] Include `action_id` in every emitted event for correlation
- [ ] Keep logging volume bounded (sampling for non-load actions)

## Phase 1 — Instrumentation (Server)

## 1.1 Realtime handler timestamps

File: `orchestrator/web/realtime_service.py`

Function blocks to instrument first:
- `msg_type == "music_load_playlist"` handler
  - capture:
    - `ws_msg_received_ts`
    - `music_load_handler_enter_ts`
    - `music_load_handler_exit_ts`
    - `ack_send_start_ts`
    - `ack_send_done_ts`
    - `state_push_scheduled_ts`
    - `state_push_done_ts`

Event names (recommended):
- `music_load.ws_received`
- `music_load.handler_enter`
- `music_load.handler_exit`
- `music_load.ack_send_start`
- `music_load.ack_send_done`
- `music_load.state_push_scheduled`
- `music_load.state_push_done`

## 1.2 Manager stage timings

File: `orchestrator/music/manager.py`

Function to instrument:
- `load_playlist()`
  - capture:
    - `load_playlist_enter_ts`
    - `list_playlists_start/end`
    - `mpd_clear_call/done`
    - `mpd_load_call/done`
    - `load_playlist_return_ts`

Event names:
- `music_load.manager_enter`
- `music_load.list_playlists_start`
- `music_load.list_playlists_done`
- `music_load.clear_start`
- `music_load.clear_done`
- `music_load.load_start`
- `music_load.load_done`
- `music_load.manager_return`

## 1.3 MPD pool queue-wait timings (critical)

File: `orchestrator/music/mpd_client.py`

Functions to instrument:
- `MPDClientPool.get_connection()`
  - capture `pool_get_request_ts`, `pool_get_grant_ts`, `pool_wait_ms`, `conn_id`
- `execute()`, `execute_list()`, `execute_batch()`
  - add `command_kind` tags (`control`, `list`, `search`, `status`, `batch`)

Event names:
- `mpd_pool.acquire_start`
- `mpd_pool.acquire_done`
- `mpd_cmd.start`
- `mpd_cmd.done`
- `mpd_cmd.error`

## 1.4 Trace writer utility

New file:
- `orchestrator/observability/latency_trace.py`

Minimal API:
- `emit(event: str, action_id: str, **fields) -> None`
- Emits JSONL to: `.openclaw/benchmarks/playlist-load/server-trace-<run_id>.jsonl`

## Phase 2 — Benchmark Harness

## 2.1 Raw MPD baseline

New file:
- `benchmarks/playlist-load/mpc_baseline.py`

Responsibilities:
- Run `mpc clear`, `mpc load <playlist>`, `mpc status`
- Repeat `--runs N`
- Output `mpc_baseline.csv`

CSV columns:
- `run_id,playlist_name,playlist_size,iter,clear_ms,load_ms,status_ms,total_ms,ok,error`

## 2.2 WS action benchmark client

New file:
- `benchmarks/playlist-load/ws_action_client.py`

Responsibilities:
- Connect to embedded WS
- Send `music_load_playlist` with unique `action_id`
- Measure:
  - send→ack
  - send→music_state update
  - send→queue update
- Output `ws_runs.csv`

CSV columns:
- `run_id,action_id,playlist_name,iter,send_ts,ack_ts,state_ts,queue_ts,ack_ms,state_ms,queue_ms,ok,error`

## 2.3 Browser E2E benchmark

New file:
- `test/selenium_playlist_load_latency.py`

Responsibilities:
- Drive real UI load action
- Measure click→ack, click→transport paint, click→queue paint
- Output `browser_runs.csv`

## 2.4 Aggregator/report

New file:
- `benchmarks/playlist-load/analyze_playlist_load.py`

Responsibilities:
- Join `mpc_baseline.csv`, `ws_runs.csv`, `browser_runs.csv`, server trace JSONL
- Compute p50/p95/p99 by stage
- Emit:
  - `summary.md`
  - `stage_breakdown.csv`
  - `outliers.csv`

## 2.5 Orchestrator runner

New file:
- `benchmarks/playlist-load/run_playlist_load_bench.py`

Responsibilities:
- One-command runner for scenarios
- Creates run folder: `benchmarks/playlist-load/results/<run_id>/`
- Invokes baseline + WS + browser tests + analyzer

## Phase 3 — Scenario Matrix

For each playlist size (`50`, `200`, `500`) and each scenario:

- [ ] `idle`
- [ ] `queue_pressure`
- [ ] `multi_client_2`
- [ ] `multi_client_4`
- [ ] `large_queue_polling`

Each combination:
- [ ] runs >= 30
- [ ] collect p50/p95/p99
- [ ] mark timeout/error rate

## Phase 4 — Decision Gates

Use these gates to choose the fix path:

- If `pool_wait_ms` contributes > 40% of p95:
  - choose **QoS split pools** (control vs list/search)
- If `ack_ms` is high while `clear/load` are low:
  - choose **early ACK + async state push**
- If queue update dominates and MPD list is slow:
  - choose **event-driven invalidation + cached queue snapshot**
- If browser paint dominates:
  - choose **UI virtualization/incremental hydration**

## Phase 5 — First Fix Candidates (ranked)

1. **Split MPD pools by command priority**
2. **Dedicated control connection (no list commands)**
3. **Event-driven queue refresh (MPD idle) over polling**
4. **Queue mirror cache service with stale-while-revalidate**

## First Command Sequence (after files above exist)

From repo root:

1) Raw MPD baseline

`python benchmarks/playlist-load/mpc_baseline.py --playlist "<name>" --size 200 --runs 30 --out benchmarks/playlist-load/results/<run_id>/mpc_baseline.csv`

2) WS benchmark

`python benchmarks/playlist-load/ws_action_client.py --ws ws://localhost:18911 --playlist "<name>" --runs 30 --out benchmarks/playlist-load/results/<run_id>/ws_runs.csv`

3) Browser benchmark

`python test/selenium_playlist_load_latency.py --base-url http://localhost:18910 --playlist "<name>" --runs 30 --out benchmarks/playlist-load/results/<run_id>/browser_runs.csv`

4) Analysis

`python benchmarks/playlist-load/analyze_playlist_load.py --run-dir benchmarks/playlist-load/results/<run_id>`

5) One-shot orchestration

`python benchmarks/playlist-load/run_playlist_load_bench.py --playlist "<name>" --sizes 50,200,500 --scenarios idle,queue_pressure,multi_client_2 --runs 30`

## Done Criteria

- [ ] Reproducible benchmark run created with a single command
- [ ] Stage-level p95 attribution clearly identifies dominant latency source
- [ ] Chosen fix validated against before/after benchmark artifacts
- [ ] p95 load UX meets agreed SLO target
