# Orchestrator Web UI Expansion - Source-Level Implementation Plan

## Scope and Constraints

This plan covers architecture and source code changes only (no runtime implementation in this step). It expands the embedded orchestrator web UI to:

- Use a modern CSS framework
- Add a persistent top-left menu dropdown
- Add an always-visible top-right microphone control button
- Add a Home page with chat/session layout backed by live websocket history
- Add a Music page showing MPD queue/current track with play/pause and click-to-play
- Show mic state colors and RMS border thickness
- Show a top header music play/pause button + current track label when music is active
- Show active timers as a bottom bar; click timer to cancel
- Mirror microphone button semantics from conference speaker (disabled/asleep/awake transitions)

It also includes explicit handling for the browser streaming overlap question:

- Browser mic streaming and orchestrator local sound-card capture must have clear authority and conflict rules


## Current Code Baseline (What Exists Today)

### Embedded Web UI service
- File: `orchestrator/web/realtime_service.py`
- Provides:
  - Inline HTML/CSS/JS in `_build_ui_html()`
  - Simple websocket status stream (`type: status`)
  - Browser audio level ingestion (`type: browser_audio_level`) and optional binary PCM chunk ingest
  - A single-page mini dashboard (not routed, no menu, no chat/music/timers pages)

### Orchestrator status + wake/sleep state
- File: `orchestrator/main.py`
- Relevant behavior:
  - Starts embedded web service when `WEB_UI_ENABLED=true`
  - Pushes periodic state via `web_service.update_orchestrator_status(...)`
  - Calls `web_service.note_hotword_detected()` on wake events
  - Wake/sleep logic already exists and is mapped to hardware play button in media-key callback:
    - Play button while awake => sleep
    - Play button while asleep => wake
  - Mute/wake/music interactions are implemented for physical controls

### Music integration
- Files:
  - `orchestrator/music/manager.py`
  - `orchestrator/music/mpd_client.py`
- Provides:
  - Playback controls (`play`, `pause`, `stop`, `play(position)`)
  - Queue retrieval (`get_queue` via `playlistinfo`)
  - Current song/status retrieval

### Timers
- Files:
  - `orchestrator/tools/timer.py`
  - `orchestrator/tools/router.py`
- Provides:
  - Active timer list (`list_active_timers`)
  - Timer cancellation by id/label

### Chat/session history source
- File: `orchestrator/main.py`
- Behavior:
  - Uses stable `session_id = config.gateway_session_prefix`
  - Quick-answer mirror can inject turns to OpenClaw session (via gateway provider), but embedded web service currently does not expose chat history over websocket.


## Target Architecture

### A. Web UI shell and routing (single embedded app)

Use a lightweight modern framework strategy in embedded UI:
- Tailwind via CDN (no build pipeline required) for immediate modernization
- Client-side hash routing (`#/home`, `#/music`) to avoid extra backend path handlers

Persistent shell:
- Top-left: menu dropdown (Home, Music)
- Top-right: microphone state button (always visible)
- Header center/right: conditional music quick-control (Play/Pause + current track)
- Main content area: page body
- Bottom fixed timer bar: active timers, click to cancel


### B. Websocket as single source of truth for live UI

Expand websocket protocol in `EmbeddedVoiceWebService`:

Server -> Client events:
- `hello` (existing, expand payload)
- `state_snapshot` (new full-state boot payload)
- `orchestrator_status` (new periodic state delta)
- `chat_history` (new initial history block)
- `chat_append` (new incremental message append)
- `music_state` (new now-playing and queue summary)
- `timers_state` (new active timers list)
- `error` (new standardized error envelope)

Client -> Server actions:
- `ui_ready`
- `mic_toggle`
- `music_toggle`
- `music_play_track` with queue position
- `timer_cancel` with timer id
- `navigate` (optional analytics/debug)
- `browser_audio_level` (existing)
- binary PCM frame (existing; optional authority-gated)


### C. Mic control semantics (conference speaker parity)

Represent UI mic state independently from wake state:
- `mic_enabled` (bool)
- `wake_state` (`asleep|awake`)
- `voice_state` (`idle|listening|sending|waiting|speaking|error`)

Button behavior state machine:
1. Initial state on web load: `mic_enabled=false` (red)
2. Click when disabled:
   - Set `mic_enabled=true`
   - Trigger wake transition (equivalent to play button wake path)
3. Click when enabled + awake:
   - Trigger sleep transition
4. Click when enabled + asleep:
   - Trigger wake transition
   - Stop music (or pause-if-playing per existing behavior policy)

Color mapping:
- Red: mic disabled
- Pink: mic enabled + asleep
- Green: mic enabled + listening/awake

RMS border thickness:
- Mic button border width = mapped function of current RMS
- Example mapping (clamped): `1px + round((rms^0.55) * 10px)`


## Detailed Source Code Change Plan

## 1) `orchestrator/web/realtime_service.py`

### 1.1 Refactor UI composition

Current: inline monolithic HTML in `_build_ui_html(ws_port)`

Planned:
- Keep embedded HTML strategy for now (no external static pipeline)
- Replace current markup with app shell + client router + component render functions
- Include Tailwind CDN script in head
- Keep websocket bootstrap as currently done

New client-side JS modules (within script block):
- `stateStore` object (single local source of truth)
- `renderShell()`
- `renderHomePage()`
- `renderMusicPage()`
- `renderTimersBar()`
- `renderMicButton()`
- `applyMicRmsBorder()`
- `connectWs()` with robust reconnect + snapshot hydration
- `dispatchAction(type, payload)` for websocket client actions

### 1.2 Add server-side state containers

Add internal fields:
- `_chat_messages: list[dict[str, Any]]`
- `_music_state: dict[str, Any]`
- `_timers_state: list[dict[str, Any]]`
- `_ui_control_state: dict[str, Any]` for `mic_enabled`, optional owner/session flags

Add update helpers:
- `update_chat_history(messages: list[dict[str, Any]])`
- `append_chat_message(message: dict[str, Any])`
- `update_music_state(**state)`
- `update_timers_state(timers: list[dict[str, Any]])`
- `update_ui_control_state(**state)`

### 1.3 Add websocket action routing

Extend `_handle_text_message()`:
- Parse action messages with `type`
- Call registered callbacks (from main) for:
  - `mic_toggle`
  - `music_toggle`
  - `music_play_track`
  - `timer_cancel`
- Validate payload shape and send `error` on invalid action

Add callback registration API:
- `set_action_handlers(...)` with callable hooks injected from `main.py`

### 1.4 Snapshot + delta broadcast model

On client connect:
- send `hello`
- send `state_snapshot` containing:
  - orchestrator status
  - ui control state
  - music state
  - timers state
  - recent chat history

Periodic loop:
- continue sending `orchestrator_status` at configured rate
- optionally send music/timers deltas when changed (hash-based change detection)


## 2) `orchestrator/main.py`

### 2.1 Wire action handlers into web service

At web service startup:
- Register async handlers bridging UI actions to existing orchestrator logic:

Handlers to add:
- `handle_ui_mic_toggle(client_id)`
  - Implements required button semantics using existing `trigger_wake` / `trigger_sleep` flow
  - Maintains `mic_enabled` state for web UI
- `handle_ui_music_toggle(client_id)`
  - Uses `music_manager.toggle_playback()`
- `handle_ui_music_play_track(position, client_id)`
  - Uses `music_manager.play(position)`
- `handle_ui_timer_cancel(timer_id, client_id)`
  - Uses `timer_manager.cancel_timer(timer_id)`

### 2.2 Share existing wake/sleep logic (avoid duplication)

Current wake/sleep code is nested inside media-key callback.

Refactor plan:
- Extract `trigger_wake(source_label)` and `trigger_sleep(source_label)` to outer shared async functions accessible by:
  - media-key callback
  - web UI action handlers
- Keep all existing side effects intact:
  - cue sounds
  - queue handling
  - music pause/restore policy
  - wake detector reset on sleep

### 2.3 Publish chat history and incremental turns to web service

Integration options:
- Primary path: whenever user transcript and assistant response are finalized in `send_debounced_transcripts` / gateway listener, also call:
  - `web_service.append_chat_message(...)`

Message format plan:
- `{ id, ts, role: 'user'|'assistant'|'system', text, source: 'native'|'browser'|'quick_answer' }`

Startup hydration:
- Optional future: load recent history from session store
- MVP in this phase: retain in-memory history from process start + live stream

### 2.4 Publish music state and timers state

Music state publisher (periodic or event-driven):
- `status = await music_manager.get_status()`
- `song = await music_manager.get_current_track()`
- `queue = await music_manager.get_queue()` (throttled, not every frame)
- Send condensed payload to web service

Timers state publisher:
- `timers = timer_manager.list_active_timers()`
- Transform to lightweight DTO including remaining seconds
- Push to web service every ~500ms or on mutate events

### 2.5 Browser stream overlap authority controls

Add runtime policy enum in main:
- `audio_input_authority = native|browser|hybrid`

Policy behavior:
- `native`: ignore browser PCM chunks (keep level telemetry only)
- `browser`: route browser PCM as active STT source, suppress local capture path
- `hybrid`: allow both but run dedupe/arbiter path (see separate plan)

Initial recommended default:
- `native`

Expose policy/state through websocket snapshot so UI can indicate active source.


## 3) `orchestrator/config.py`

Add config fields:
- `web_ui_css_framework: str = "tailwind-cdn"`
- `web_ui_chat_history_limit: int = 200`
- `web_ui_music_poll_ms: int = 1000`
- `web_ui_timer_poll_ms: int = 500`
- `web_ui_mic_starts_disabled: bool = True`
- `web_ui_audio_authority: str = "native"`  # native|browser|hybrid

Validate:
- authority enum
- poll intervals > minimum threshold


## 4) `orchestrator/music/manager.py`

No breaking API changes required, but add thin helpers for UI efficiency:
- `get_ui_music_state()` returns compact state:
  - playback state, elapsed, duration, queue length, current track meta
- `get_ui_playlist(limit: int = 200)` returns queue DTO with position/title/artist/album/file


## 5) `orchestrator/tools/timer.py`

No core logic rewrite needed.

Add helper for UI DTO:
- `to_ui_dict(now_ts)` method on `Timer` or mapper in manager:
  - id, label, remaining_seconds, expires_at


## 6) New documentation update

Update existing docs:
- `EMBEDDED_WEB_UI.md`

Add sections:
- New routes/pages (`#/home`, `#/music`)
- New websocket events/actions
- Mic state machine behavior
- Browser/native audio authority explanation


## UX Behavior Details (Source-Driven)

### Header behavior
- Left: menu dropdown always visible
- Right: mic button always visible
- Center/right additional control appears only when MPD state is `play` or `pause` with active current track

### Home page
- Chat-like layout from local state store
- Websocket drives append updates
- Reconnect logic requests full snapshot to avoid divergence

### Music page
- Renders queue rows from `music_state.queue`
- Click row => send `music_play_track` with position
- Play/Pause button => send `music_toggle`

### Timer bar
- Fixed bottom container
- Each timer chip shows label + mm:ss
- Click chip => `timer_cancel`
- Timer bar hidden when no active timers


## Streaming Overlap Clarification (Design Position)

Question answered for implementation:
- Browser streaming and direct sound-card capture are currently parallel-capable but not explicitly arbitrated.
- The expanded UI should not silently merge both by default.
- Default policy: native capture is authoritative; browser stream contributes telemetry unless explicitly switched.

Operational impact:
- Prevents duplicate transcripts and accidental double-trigger from same speech across two inputs.


## Incremental Implementation Phases (Execution Order)

1. Protocol + state foundation in `realtime_service.py`
2. Shared wake/sleep handlers extraction in `main.py`
3. UI shell replacement + routing + mic button behavior
4. Music state feed + music page actions
5. Timer feed + bottom cancellation bar
6. Chat history live feed and append pipeline
7. Audio authority guardrails and docs update


## Acceptance Criteria Checklist

- [ ] UI uses modern CSS framework and retains embedded deployment simplicity
- [ ] Top-left menu dropdown present and functional
- [ ] Top-right mic button always visible and follows required state machine
- [ ] Mic colors map exactly: red disabled, pink asleep, green listening
- [ ] Mic RMS visibly adjusts button border thickness
- [ ] Home page shows current session chat and updates live over websocket
- [ ] Music page shows queue/current track and supports play/pause + click-to-play
- [ ] Header music quick-control appears only when music context exists
- [ ] Bottom timer bar shows active timers and supports click-to-cancel
- [ ] Browser/native overlap behavior is explicit, configurable, and documented


## Risks and Mitigations

- Risk: Nested wake/sleep logic in current callback makes reuse error-prone
  - Mitigation: extract shared async functions before adding web handlers

- Risk: Polling queue too frequently can load MPD
  - Mitigation: throttle queue refresh and use hash-based dedupe before websocket emit

- Risk: Chat order race between quick-answer and gateway streaming
  - Mitigation: include monotonic sequence numbers in appended chat messages

- Risk: Multiple UI clients issuing conflicting actions
  - Mitigation: action ack payload includes origin and resulting canonical state; optional controller lease in phase 2


## Out of Scope for This Plan

- Full frontend build system migration (React/Vite)
- Authentication/authorization model for remote UI exposure
- Cross-process transcript arbiter implementation details (handled in separate plan document)
