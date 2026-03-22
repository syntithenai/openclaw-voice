# Voice Overlap and Multi-Source Debounce - Independently Executable Plan

## Objective

Provide a standalone, deployable path to prevent duplicate transcript actions when audio arrives from:

- Multiple orchestrators in the same room
- Multiple browser sessions
- Mixed browser streaming + native sound-card capture on the same orchestrator

This plan is intentionally independent from web UI expansion and can be executed on its own.


## Deliverable

A transcript arbitration layer that:
- Deduplicates near-identical utterances within a configurable time window
- Elects one authoritative command for execution
- Preserves observability of all candidate transcripts for debugging
- Remains safe under multi-client and multi-device concurrency


## Execution Boundaries

- No frontend dependency required
- Can be enabled in native orchestrator-only deployments
- Controlled by config flag: rollout as opt-in first


## Core Design

## 1) Introduce normalized transcript event envelope

Add new internal event DTO (new module):
- `orchestrator/transcript_arbiter/types.py`

Fields:
- `event_id` (uuid)
- `source_id` (orchestrator instance id)
- `input_id` (native mic id or browser client id)
- `session_id`
- `ts_mono`
- `ts_wall`
- `text_raw`
- `text_norm`
- `confidence` (if available)
- `emotion` (optional)
- `audio_rms`
- `wake_state_snapshot`

Normalization pipeline:
- lowercasing
- punctuation fold
- whitespace collapse
- number canonicalization (optional phase 2)


## 2) Add arbitration service

New module:
- `orchestrator/transcript_arbiter/service.py`

Primary class:
- `TranscriptArbiter`

Responsibilities:
- Maintain short-lived sliding window of recent accepted/rejected events
- Compute duplicate likelihood against recent accepted events
- Decide one of:
  - `ACCEPT`
  - `REJECT_DUPLICATE`
  - `REJECT_LOW_PRIORITY`
  - `DEFER_WAIT_FOR_WINDOW`


## 3) Matching and debounce algorithm

### 3.1 Time window
- `window_ms` default 1200ms for same-room overlap

### 3.2 Similarity checks
Use tiered checks for speed:
1. Exact normalized text match
2. Token-set similarity (Jaccard)
3. Edit distance ratio (Levenshtein normalized)
4. Optional phonetic key match (Metaphone/Soundex) phase 2

### 3.3 Acceptance scoring
Score components:
- Similarity confidence
- Source priority
- Signal quality (RMS/confidence)
- Freshness

Accept highest-scoring candidate within the window; suppress others.


## 4) Source priority policy

Add deterministic priority table in config:
- `native_local_mic` > `hardware_button_wake_path` > `browser_stream`
- Remote orchestrators can be weighted by room role / device placement

Config example fields in `orchestrator/config.py`:
- `transcript_arbiter_enabled: bool = False`
- `transcript_arbiter_window_ms: int = 1200`
- `transcript_arbiter_similarity_threshold: float = 0.88`
- `transcript_arbiter_source_priority: str` (json mapping)
- `transcript_arbiter_emit_debug_events: bool = True`


## 5) Integration points in existing code

### 5.1 `orchestrator/main.py`

At transcript finalization path (`process_chunk` / `send_debounced_transcripts`):
- Wrap transcript into arbiter event
- Ask arbiter for decision before enqueuing gateway request

Behavior:
- Accepted event proceeds to normal debounce/gateway pipeline
- Rejected duplicate logs reason and increments metric counter

### 5.2 Browser websocket path

In `orchestrator/web/realtime_service.py` + main handler glue:
- Tag browser-origin transcripts with `input_id=ws_client_id`
- Pass to same arbiter service

### 5.3 Multi-orchestrator federation (optional independent sub-phase)

Add lightweight peer gossip endpoint (HTTP/WS) for accepted-event fingerprints only:
- Share hash, timestamp, and source metadata
- Avoid sharing full audio

This allows room-wide duplicate suppression without centralizing raw audio.

### 5.4 LAN discovery and federation transport (detailed)

This is the hard part. Use a layered approach so discovery still works in imperfect LANs:

#### Discovery strategy (in priority order)

1. **mDNS / DNS-SD (primary)**
   - Service type: `_openclaw-arbiter._tcp.local`
   - Each node advertises:
     - `instance_id`
     - `cluster_id` (room/group name)
     - `api_port`
     - `proto=v1`
     - `priority` (optional static weight)
   - Each node browses same service type and keeps a live peer table.

2. **UDP beacon fallback (for networks where mDNS is filtered)**
   - Broadcast/multicast heartbeat every 2s on configurable port
   - Payload: signed minimal node descriptor (`instance_id`, `cluster_id`, `api_port`, `epoch`)
   - Receiver verifies signature then probes peer via HTTP `/arbiter/hello`.

3. **Static peers fallback (manual list)**
   - Config: `transcript_federation_static_peers=["192.168.1.10:18960", ...]`
   - Used when both mDNS and UDP are unavailable (enterprise VLANs / AP isolation).

#### New config fields (for `orchestrator/config.py`)

- `transcript_federation_enabled: bool = False`
- `transcript_federation_cluster_id: str = "default-room"`
- `transcript_federation_discovery_mode: str = "mdns"`  # mdns|udp|static|hybrid
- `transcript_federation_api_host: str = "0.0.0.0"`
- `transcript_federation_api_port: int = 18960`
- `transcript_federation_udp_port: int = 18961`
- `transcript_federation_heartbeat_ms: int = 2000`
- `transcript_federation_peer_ttl_ms: int = 9000`
- `transcript_federation_static_peers: str = ""`  # comma-separated host:port
- `transcript_federation_shared_secret: str = ""`  # required unless allow_insecure=true
- `transcript_federation_allow_insecure: bool = False`

#### Required new modules

- `orchestrator/transcript_arbiter/federation/discovery.py`
  - `MdnsDiscovery`, `UdpDiscovery`, `StaticDiscovery`
- `orchestrator/transcript_arbiter/federation/transport.py`
  - HTTP/WS client for peer gossip publish/subscribe
- `orchestrator/transcript_arbiter/federation/registry.py`
  - In-memory peer registry with TTL expiry and health state
- `orchestrator/transcript_arbiter/federation/server.py`
  - Local endpoints for peer handshake and fingerprint ingestion

#### Peer lifecycle flow

1. Node starts local federation server (`:18960` default).
2. Node announces itself via discovery mode.
3. Node receives peer candidate, performs handshake:
   - `GET /arbiter/hello`
   - Validate `cluster_id`, `proto`, signature/HMAC.
4. On success, peer enters `ACTIVE` state and receives gossip events.
5. If no heartbeat/traffic for `peer_ttl_ms`, mark `STALE` then evict.

#### Federation API contract (minimal)

- `GET /arbiter/hello`
  - Returns node descriptor and nonce challenge.
- `POST /arbiter/publish`
  - Body: accepted transcript fingerprint event (no raw transcript/audio).
- `GET /arbiter/peers`
  - Local debug endpoint for current registry state.
- Optional `WS /arbiter/stream`
  - Push channel for low-latency gossip instead of polling.

Fingerprint publish payload:
- `event_id`
- `source_id`
- `cluster_id`
- `ts_wall_ms`
- `fingerprint` (sha256 of normalized text + intent salt)
- `similarity_hint` (optional)
- `priority`
- `signature`

#### Security model (LAN-safe baseline)

- Every payload signed with HMAC-SHA256 using `transcript_federation_shared_secret`.
- Reject unsigned payloads unless `allow_insecure=true`.
- Include `ts_wall_ms` + nonce to prevent replay.
- Reject events outside skew window (e.g., ±5s) unless in observe mode.
- Scope federation strictly by `cluster_id` to avoid cross-home bleed.

#### Conflict resolution across peers

No global leader required (leaderless deterministic resolution):
- For equivalent events in same arbitration window, choose winner by:
  1. Highest source priority
  2. Better confidence/RMS
  3. Earliest timestamp
  4. Lexicographically smallest `source_id` (final tie-break)

All nodes apply same rule => same winner decision without coordinator.

#### Handling common LAN failure modes

- **mDNS blocked**: auto-fallback to UDP/static in `hybrid` mode.
- **AP client isolation**: static peers only (same subnet direct routes required).
- **Clock drift**: prefer monotonic local windows + generous wall-clock skew tolerance.
- **Network partition**: each partition arbitrates locally; idempotency still prevents rapid duplicate actions.
- **Peer flapping**: require N consecutive misses before eviction; exponential backoff reconnect.

#### Execution order for federation implementation

1. Implement local federation server (`hello`, `publish`, peer registry).
2. Add static peer mode (simplest deterministic path).
3. Add mDNS discovery.
4. Add UDP fallback + `hybrid` mode.
5. Add WS stream optimization (optional; HTTP publish is enough for MVP).

#### Independent acceptance criteria for discovery/federation

- [ ] Two orchestrators auto-discover via mDNS on same LAN and exchange fingerprints.
- [ ] With mDNS disabled, static peer mode still deduplicates correctly.
- [ ] Invalid signature payloads are rejected and logged.
- [ ] Peer TTL expiry removes offline nodes without process restart.
- [ ] Deterministic winner selection matches across all active peers.


## 6) Concurrency and command idempotency

Introduce command idempotency key:
- Key basis: accepted transcript fingerprint + action intent + coarse timestamp bucket

Before executing tool/music/wake command:
- Check idempotency cache (TTL 2-5s)
- Drop duplicates already executed

This protects against race from:
- Multiple browser sessions
- Retry loops
- concurrent orchestrators


## 7) Observability and diagnostics

Add counters/logging:
- `arbiter.accepted_total`
- `arbiter.duplicate_rejected_total`
- `arbiter.priority_rejected_total`
- `arbiter.deferred_total`
- decision latency histogram

Optional event log file:
- `timers/events` style JSONL stream for arbiter decisions
- Include reason codes and similarity scores


## 8) Test Plan (Independent)

Unit tests (new folder `orchestrator/transcript_arbiter/tests`):
- Exact duplicate within window => one accept
- Near-duplicate within threshold => one accept
- Same text outside window => accept again
- Lower-priority source rejected when higher-priority source equivalent exists
- Idempotency cache prevents duplicate command execution

Integration tests:
- Simulated dual-source same utterance (native + browser)
- Simulated two orchestrators sending same transcript with 100-500ms skew
- Rapid websocket reconnect replay does not duplicate command dispatch


## 9) Rollout Plan

Phase A (dark launch):
- Arbiter runs in observe-only mode
- Logs decisions but does not suppress

Phase B (soft enforce):
- Enforce only exact-match suppression

Phase C (full enforce):
- Enable similarity-based suppression + source priority

Phase D (federated optional):
- Enable cross-orchestrator fingerprint sharing


## 10) Safety Defaults

- If arbiter fails, fallback to existing behavior (fail-open) with explicit warning
- Config guards prevent accidental hard lock of transcripts
- Keep thresholds conservative initially to avoid dropping valid speech


## Independent Acceptance Criteria

- [ ] Same utterance from two local inputs results in one command execution
- [ ] Same utterance from two orchestrators in overlap window results in one command execution (when federation enabled)
- [ ] Different utterances are not incorrectly collapsed
- [ ] Replay/reconnect does not duplicate tool/music actions
- [ ] Debug logs clearly explain why candidates were accepted/rejected


## Minimal Initial Implementation Cut (Executable MVP)

If implementing in smallest safe slice first:
1. Add arbiter module with exact-normalized-text + time-window suppression only
2. Integrate at `send_debounced_transcripts` path in `main.py`
3. Add idempotency cache for command dispatch
4. Add counters/logging
5. Enable with `transcript_arbiter_enabled=true`

This MVP already solves most duplicate-action pain and can be deployed independently of UI work.
