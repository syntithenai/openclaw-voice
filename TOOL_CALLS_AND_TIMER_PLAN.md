# Tool Calls Integration for Fast LLM Response - Timer/Alarm Skill Plan

**Created:** March 9, 2026  
**Project:** OpenClaw Voice Orchestrator  
**Focus:** Quick Response Latency with Tool Integration

---

## Executive Summary

This document outlines a strategy to incorporate tool call capabilities into the LLM fast response system, starting with a timer/alarm clock skill. The goal is to enable the orchestrator to immediately respond to common timer/alarm queries without gateway round-trips while maintaining long-term scalability for additional tool-driven skills.

### Important Design Guardrails
- Keep timer/alarm persistence independent from gateway availability.
- Use deterministic fast-path parsing for obvious timer/alarm commands before LLM fallback.
- Use UUIDv7 for all timer/alarm identifiers (sortable, collision-safe).
- Implement write debouncing for alarm state updates to minimize disk I/O.
- Treat file writes as critical path: atomic write, fsync where needed, and corruption quarantine.
- Keep timer state minimal and ephemeral; timers are deleted on completion as requested.

---

## Current Architecture Overview

### Voice Orchestrator Pipeline
```
Audio Input → VAD → STT (Whisper) → LLM (Quick Answer) → TTS (Piper) → Audio Output
                                         ↓
                                    Gateway (if needed)
```

### Quick Answer System
- **File:** `orchestrator/gateway/quick_answer.py`
- **Purpose:** Provide immediate factual answers before escalating to gateway
- **Current Limitation:** Text-only responses; no ability to execute actions or maintain state
- **Implementation:** Uses OpenAI-compatible chat completions API with strict system prompt

### Gateway System
- **File:** `orchestrator/gateway/client.py`
- **Providers:** OpenClaw, ZeroClaw, TinyClaw, IronClaw, MimiClaw, PicoClaw, NanoBot
- **Current Role:** Handle complex reasoning and multi-turn conversations

---

## Proposed Design: Tool-Enabled Quick Answer System

### Core Concept

Extend the QuickAnswerClient to support **tool calls** (function calling), enabling the LLM to:
1. **Recognize** when a user request requires a tool execution
2. **Request** the execution of specific tools with parameters
3. **Receive** tool results and incorporate them into responses

### Architecture

```
┌─────────────────────────────────────────────────────┐
│          Voice Orchestrator (Main)                  │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Quick Answer LLM  │  (Extended with tools)
         └─────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │   Tool Router       │  (New)
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
[Timer Tool]  [Alarm Tool]  [Other Tools...]
    │              │              │
    └──────────────┴──────────────┘
             │
    ┌────────▼─────────┐
    │  State Manager   │  (New)
    │  - Active timers │
    │  - Alarms        │
    │  - History       │
    └──────────────────┘
```

### Tool Execution Flow

1. **User Request** → STT produces transcript (e.g., "Set a timer for 10 minutes")
2. **LLM Tool Call** → LLM recognizes intent and requests tool execution
3. **Tool Execution** → Router executes tool (set timer in background)
4. **Immediate Response** → LLM generates spoken response (e.g., "Timer set for 10 minutes")
5. **State Persistence** → Timer stored in state manager for background processing

---

## Phase 1: Timer Skill Implementation

### 1.1 Core Timer Functionality

**File:** `orchestrator/tools/timer.py`

#### Features
- **Set Timer:** `set_timer(duration_seconds: int, label: str = "") → timer_id`
- **Cancel Timer:** `cancel_timer(timer_id: str) → success`
- **List Timers:** `list_active_timers() → List[Timer]`
- **Check Timer:** `get_timer_status(timer_id: str) → Timer`

#### Timer Data Structure
```python
@dataclass
class Timer:
    id: str                      # Unique identifier
    duration_seconds: int        # Original duration
    created_at: float            # Unix timestamp
    label: str                   # User-provided name (e.g., "laundry")
    expires_at: float            # Calculated expiration time
    completed: bool              # Whether timer finished
    cancelled: bool              # Whether user cancelled it
    callback: Optional[Callable] # Notification handler
```

#### Notification Behavior
- **On Expiration:** Single alert sequence
  1. Play bell sound (distinct audio cue)
  2. Speak the timer name (e.g., "Laundry timer done") then stop
  3. Log notification for monitoring
  4. Delete the timer file from `timers/` after completion
- **Interruption:** Any voice activity (cut-in) stops playback immediately
- **Persistence:** Stored as one file per timer in workspace `timers/` folder

### 1.2 Alarm Clock Functionality

**File:** `orchestrator/tools/alarm.py`

#### Features
- **Set Alarm:** `set_alarm(trigger_time: str, label: str = "") → alarm_id`
  - Input: "6:30 AM" or "18:30" or "in 2 hours"
- **Cancel Alarm:** `cancel_alarm(alarm_id: str) → success` (prevent future alarm)
- **Stop Alarm:** `stop_alarm(alarm_id: str | None = None) → success` (stop actively ringing alarm; if omitted, stop all ringing alarms)
- **List Alarms:** `list_alarms() → List[Alarm]`
- **Get Alarm:** `get_alarm_status(alarm_id: str) → Alarm`

#### Alarm Data Structure
```python
@dataclass
class Alarm:
    id: str                      # Unique identifier
    trigger_time: datetime       # When alarm should trigger
    label: str                   # User-provided name
    created_at: float            # Unix timestamp
    enabled: bool                # Active/inactive
    triggered: bool              # Has alarm fired?
    callback: Optional[Callable] # Notification handler
```

#### Notification Behavior
- **At Trigger Time:** Continuous alert loop
  1. Play bell sound repeatedly (distinct audio cue) every 1-2 seconds
  2. Continue until explicitly stopped by voice command ("stop alarm", "dismiss") or cut-in
  3. Mark `triggered=TRUE` in state store
  4. Keep `notified=FALSE` until user stops the alarm
- **Stop/Dismiss:** Voice command stops the alert immediately and marks as notified
  - With no alarm specified: stop **all** currently ringing alarms
  - With label/id specified: stop only matching alarm(s)
- **Snooze Option:** Allow user voice commands ("snooze 5 minutes") to pause and resume
- **Cut-in:** Any speech activity automatically stops alarm and triggers transcript processing
- **Persistence:** Triggered alarms survive restarts and continue ringing on resume

### 1.3 Tool Call Schema (OpenAI Function Calling Format)

```json
{
  "type": "function",
  "function": {
    "name": "set_timer",
    "description": "Set a countdown timer",
    "parameters": {
      "type": "object",
      "properties": {
        "duration_seconds": {
          "type": "integer",
          "description": "Duration in seconds (e.g., 600 for 10 minutes)"
        },
        "label": {
          "type": "string",
          "description": "Optional name for the timer (e.g., 'laundry', 'cooking')"
        }
      },
      "required": ["duration_seconds"]
    }
  }
}
```

---

## Phase 2: Tool-Aware Quick Answer System

### 2.1 Extended LLM System Prompt

```
You are a voice assistant with access to real-time tools. Your role is to:

1. IMMEDIATELY recognize when a request requires tool execution
2. Call the appropriate tool with correct parameters
3. Use tool results to formulate a spoken response
4. Keep responses brief (1-2 sentences for voice)

Available Tools:
- set_timer(duration_seconds, label=""): Set a countdown timer
- cancel_timer(timer_id): Cancel an active timer
- list_timers(): Get all active timers
- set_alarm(trigger_time, label=""): Set an alarm
- cancel_alarm(alarm_id): Cancel a scheduled alarm
- stop_alarm(alarm_id=None): Stop a ringing alarm; when omitted, stop all ringing alarms
- list_alarms(): Get all alarms

Instructions:
- For timer/alarm requests: ALWAYS use the appropriate tool
- Parse user time expressions (e.g., "10 minutes", "6:30 AM", "in 20 mins")
- Return tool results naturally in spoken responses
- If parsing fails, ask for clarification rather than guess

Current date/time: {current_datetime}
```

### 2.2 Enhanced QuickAnswerClient

**File:** `orchestrator/gateway/quick_answer.py` (extended)

#### Changes
1. **Tool Definition Registry:** Load available tools at initialization
2. **Tool Call Detection:** Parse LLM response for `tool_calls` when using function calling
3. **Synchronous Tool Execution:** Execute tools before generating final response
4. **Result Integration:** Pass tool results back to LLM for response generation

#### Pseudocode
```python
async def get_quick_answer_with_tools(self, user_query: str):
    # Step 1: Initial LLM call with tools available
    initial_response = await self.llm_call(
        system_prompt=SYSTEM_PROMPT_WITH_TOOLS,
        messages=[{"role": "user", "content": user_query}],
        tools=self.available_tools,
    )
    
    # Step 2: Check for tool calls
    if initial_response.tool_calls:
        tool_results = []
        for tool_call in initial_response.tool_calls:
            result = await self.execute_tool(tool_call)
            tool_results.append({
                "tool_call_id": tool_call.id,
                "result": result
            })
        
        # Step 3: Second LLM call with tool results
        final_response = await self.llm_call(
            messages=[
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": initial_response.content},
                {"role": "tool", "content": tool_results}
            ]
        )
        return final_response.text
    else:
        return initial_response.text
```

### 2.3 Tool Router

**File:** `orchestrator/tools/router.py`

```python
class ToolRouter:
    def __init__(self):
        self.tools = {
            "set_timer": self.set_timer,
            "cancel_timer": self.cancel_timer,
            "list_timers": self.list_timers,
            "set_alarm": self.set_alarm,
            "cancel_alarm": self.cancel_alarm,
      "stop_alarm": self.stop_alarm,
            "list_alarms": self.list_alarms,
        }
    
    async def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return result"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = await self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except ValueError as e:
            return {"error": str(e)}
```

#### State Tracking for Alarm Alerts
- **Ringing State:** Track which alarm is currently ringing (if any)
- **Stop Signal:** When `stop_alarm()` called, immediately halt alert playback
- **Automatic Stop on Cut-in:** Voice activity during alarm detected by VAD triggers implicit stop + transcript processing
- **Default Behavior:** `stop_alarm()` with no id/label stops all currently ringing alarms
- **Targeted Behavior:** User can say "stop [label] alarm" to stop specific alarm(s)

---

## Phase 3: State Management

### 3.1 Persistent File System Store

**File:** `orchestrator/tools/state.py`

#### Requirements
- **Backend:** File-per-entity JSON persistence (no SQL database)
- **Location:** `<workspace_root>/timers/`
- **Durability:** Write each timer/alarm to its own file immediately on create/update
- **Recovery:** Rehydrate active timers/alarms by scanning `timers/` on startup
- **Thread Safety:** Use asyncio locks for concurrent access
- **Atomic Writes:** Write temp file + atomic rename

#### Initialization Flow
```python
async def initialize():
    # On startup:
    # 1. Ensure <workspace_root>/timers exists
    # 2. Load all timer-*.json files and alarm-*.json files
    # 3. Validate state (remove stale/corrupt files, skip invalid payloads)
    # 4. Resume monitoring of loaded timers/alarms
    # 5. Emit TTS notification of recovered state (e.g., "I found a laundry timer with 5 minutes left")
```

#### File Format
```text
timers/
  active/
    timer-<id>.json
    alarm-<id>.json
  events/
    events-YYYY-MM-DD.jsonl   # optional append-only audit log
  quarantine/
    *.json                    # malformed/corrupt files moved here
  .index.json                 # optional optimization for quick listing
```

#### File Naming Convention
- **Required:** Use UUIDv7 for all timer/alarm IDs
- Format: `timer-01h2xcejqtf2nbrexx3vqjhp41.json`
- Benefits:
  - Timestamp prefix allows chronological sorting
  - No collision risk across processes/restarts
  - Easy debugging (IDs sort by creation time)
  - 128-bit entropy ensures uniqueness
- Implementation: Use Python `uuid6` or `uuid-utils` library

```json
// timer-<id>.json
{
  "schema_version": 1,
  "type": "timer",
  "id": "01h2xcejqtf2nbrexx3vqjhp41",
  "name": "laundry",
  "duration_seconds": 600,
  "created_at": 1741516800.0,
  "expires_at": 1741517400.0,
  "cancelled": false
}
```

```json
// alarm-<id>.json
{
  "schema_version": 1,
  "type": "alarm",
  "id": "z9y8x7",
  "name": "bedroom",
  "trigger_time": 1741520400.0,
  "created_at": 1741516800.0,
  "enabled": true,
  "triggered": false,
  "ringing": false
}
```

#### Gateway Restart Resilience
- **On Restart:** Load all non-cancelled, non-complete timers with remaining time
- **Expired Timers:** If timer was supposed to expire during downtime, alert immediately
- **Paused Alarms:** If alarm was supposed to trigger, trigger now or skip based on threshold
- **User Notification:** "Welcome back! You have an active 3-minute laundry timer"
- **Cleanup Rule:** Delete timer file when timer completes or is cancelled; keep alarm files until cancelled/disabled
- **Clock Safety:** Store UTC epoch timestamps only; never local-formatted times in persisted files

### 3.2 Background Monitoring

**File:** `orchestrator/tools/monitor.py`

#### Responsibilities
- **Timer Expiration Detection:** Check active timers every 100ms, compare against expires_at
- **Alarm Triggering:** Monitor alarms for trigger_time match; start continuous ring loop
- **Alarm Ring Loop:** Keep playing bell sound + repeating every 1-2 seconds until `stop_alarm()` or cut-in detected
- **State Persistence:** Write checkpoint to disk on state changes
- **Alert Playback:** Generate audio alerts (bell sound) for timers (once) and alarms (looping)
- **TTS Integration:** Speak timer label once on expiration; don't speak during alarm loop
- **Cut-in Detection:** Monitor VAD; if speech detected during alarm, auto-stop alert
- **Clean-up:** Delete completed/cancelled timer files; optionally append completion event to `events/*.jsonl`
- **Graceful Shutdown:** Flush all pending state to disk before exit; stop any active alerts

#### Write Debouncing for Alarm States
- **Problem:** Alarm `ringing` state can toggle rapidly (every 1-2 seconds during bell loop)
- **Solution:** Coalesce rapid state changes within a 50-100ms window before writing to disk
- **Implementation:**
  - Buffer state changes in memory with timestamp
  - Flush buffered writes after debounce window expires
  - Always flush immediately on graceful shutdown or critical state changes
  - Only debounce non-critical fields (`ringing`); write `triggered`, `enabled` immediately
- **Benefit:** Reduces disk I/O contention by ~90% during active alarm playback
- **Trade-off:** Up to 100ms of `ringing` state could be lost on crash (acceptable for this field)

---

## Phase 4: Integration with Main Orchestrator

### 4.1 Initialization

**File:** `orchestrator/main.py` (modified)

```python
# In VoiceOrchestrator.__init__():
self.tool_router = ToolRouter()
self.tool_monitor = ToolMonitor(tool_router)
self.state_manager = StateManager()

# Start background monitoring
asyncio.create_task(self.tool_monitor.run())
```

### 4.2 STT → Quick Answer Flow

**File:** `orchestrator/main.py` (modified)

```python
# Current flow:
transcript = await whisper_client.transcribe(audio)
should_escalate, response = await quick_answer_client.get_quick_answer(transcript)

# New flow with tools:
transcript = await whisper_client.transcribe(audio)

# Optional: Deterministic fast-path for obvious timer/alarm commands
# Parse simple patterns first ("set timer 5 min", "stop alarm") to bypass LLM latency
fast_path_result = await self.tool_router.try_deterministic_parse(transcript)
if fast_path_result:
    response = fast_path_result.response
    await piper_client.synthesize(response)
else:
    # Fallback to LLM with tool calling
    should_escalate, response = await quick_answer_client.get_quick_answer_with_tools(
        transcript,
        tool_router=self.tool_router
    )
    
    if not should_escalate:
        # Immediate response (may have triggered tool calls)
        await piper_client.synthesize(response)
    else:
        # Gateway escalation (if needed)
        response = await gateway_client.send_transcript(transcript)
        await piper_client.synthesize(response)
```

#### Deterministic Fast-Path Parsing

**Required:** Implement pattern-matching fast path before LLM for latency-critical timer/alarm commands:

- **Target:** Reduce latency to <200ms for obvious commands
- **Patterns:** Regex-based detection for common phrases:
  - `set timer (\d+) (min|minute|sec|second|hour)s?` → direct tool call
  - `set (a |an )?(\d+) (min|minute|sec|second|hour) timer` → direct tool call
  - `stop( all)? alarm(s)?` → direct stop_alarm() call
  - `cancel (\w+) timer` → direct cancel by label
  - `cancel( all)? timer(s)?` → direct cancel call
  - `(how much time|time left|time remaining)` → direct list/status query
- **Fallback:** If pattern doesn't match or parsing fails, use LLM tool calling
- **Benefit:** Shaves 100-200ms LLM inference time for ~70-80% of timer/alarm requests
- **Implementation File:** `orchestrator/tools/parser.py`

---

## Phase 5: Spoken Interaction Extensions

### 5.1 Timer State Commands

Users should be able to query timer state via voice:
- "How much time is left on my timer?" → Query active timer
- "What timers do I have?" → List all timers
- "Cancel the laundry timer" → Cancel specific timer by label
- "Cancel all timers" → Cancel everything

### 5.2 Alarm Control Commands

For alarms:
- "Stop alarm" → Stop all currently ringing alarms (default when none specified)
- "Dismiss alarm" → Dismiss/acknowledge alarm
- "Stop [label] alarm" → Stop specific alarm by name (e.g., "stop bedroom alarm")
- "Snooze" or "Snooze for 5 minutes" → Pause and resume after duration
- **Cut-in:** Speaking during alarm automatically stops it

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Implement `orchestrator/tools/timer.py` with basic set/cancel/list
- [ ] Implement `orchestrator/tools/alarm.py` with time parsing
- [ ] Create `orchestrator/tools/state.py` with file-per-item JSON backend in workspace `timers/`
- [ ] Implement UUIDv7 ID generation for all timers/alarms
- [ ] Implement write debouncing for alarm state updates
- [ ] Write unit tests for timer/alarm logic

### Week 2: Tool Router & Integration
- [ ] Implement `orchestrator/tools/router.py` with tool registry (set/cancel/stop tools)
- [ ] Implement `orchestrator/tools/parser.py` with deterministic fast-path parsing
- [ ] Integrate fast-path parser into main orchestrator flow (before LLM)
- [ ] Extend `QuickAnswerClient` to support OpenAI function calling
- [ ] Implement tool call detection and execution flow
- [ ] Add stop_alarm handler to tool router
- [ ] Write integration tests

### Week 3: Background Monitoring & Notifications
- [ ] Implement `orchestrator/tools/monitor.py` for expiration detection
- [ ] Implement alarm triggering with continuous bell loop
- [ ] Implement stop_alarm handler (cease bell immediately)
- [ ] Create audio alert generation (distinct bell sound vs TTS)
- [ ] Add VAD-based cut-in detection during alarm playback (auto-stop)
- [ ] Test snooze functionality
- [ ] Test timer single-shot vs alarm continuous loop behavior

### Week 4: Voice Commands & Polish
- [ ] Add spoken timer/alarm query support
- [ ] Test end-to-end flows (voice → tool → response)
- [ ] Performance optimization (latency targets: <500ms for tool execution)
- [ ] Documentation and examples

---

## Technical Considerations

### Latency Budget
- **Target:** <500ms from tool call to spoken response
- **Fast-Path Target:** <200ms for deterministic commands (bypassing LLM)
- **LLM Fallback Breakdown:**
  - LLM inference + tool call generation: 200ms
  - Tool execution: 50ms
  - TTS synthesis: 200ms
  - Audio playback: 50ms
- **Fast-Path Breakdown:**
  - Pattern matching + parsing: 10ms
  - Tool execution: 50ms
  - TTS synthesis: 200ms
  - Audio playback: 50ms (saves ~190ms)

### Concurrency
- Multiple timers/alarms running simultaneously
- Voice commands during timer/alarm state
- Solution: Use asyncio locks for state access, non-blocking timer checks

### File System Persistence
- **Location:** `<workspace_root>/timers/`
- **Format:** One JSON file per timer/alarm (`timer-*.json`, `alarm-*.json`)
- **On Restart:**
  1. Scan `timers/active/` for timer/alarm files
  2. Load all non-cancelled timers with remaining time calculated
  3. Load all enabled alarms with upcoming trigger times
  4. Validate malformed/stale files, reconnect valid entries to monitoring
  5. Notify user of recovered state via TTS (e.g., "Resuming kitchen timer with 5 minutes left")
- **Downtime Handling:** 
  - If timer expires during downtime: Alert immediately on startup (play audio + TTS)
  - If alarm was missed: Fire alert if within 1 hour window, skip if older
- **Cleanup:** Delete timer files when completed/cancelled; keep alarm files until cancelled/disabled
- **Graceful Shutdown:** Flush all pending file writes before exit

### Error Handling
- Invalid time expressions → Ask user for clarification
- Tool execution failure → Fall back to gateway or error response
- File I/O errors → Log warning, use in-memory cache, retry on next state change
- Missing `timers/` folder on startup → Create it
- Corrupted timer/alarm file → Move to `timers/quarantine/` and continue
- Duplicate file ids or partial writes → Recover from latest valid file and rewrite canonical form

### Voice Interaction Constraints
- Keep responses to 1-2 sentences
- Use clear, unambiguous spoken feedback
- Support natural time expressions (10 min, 6:30 AM, "quarter past 3")

---

## Future Extensions (Post-MVP)

### Additional Tools (Phase N)
1. **Notes/Reminders:** Create, list, delete voice notes
2. **Weather:** Current conditions, forecasts (requires integration)
3. **Calculations:** Math operations ("What's 15% of 200?")
4. **Unit Conversion:** Distance, temperature, weight conversions
5. **Scheduling:** Integration with calendar APIs

### Advanced Features
- **Context Memory:** Remember previous timer labels for quick re-creation
- **Predictive Triggers:** "Set another timer for the same duration"
- **Multi-language Support:** Parse time expressions in multiple languages
- **Low-power Mode:** Reduce monitoring frequency on battery systems

### Analytics
- Track tool usage patterns
- Monitor latency metrics
- Identify common user requests for future tools

---

## Testing Strategy

### Unit Tests
- Timer duration calculations and edge cases (0s, 1us, 999999s)
- Time expression parsing ("in 5 mins", "at 3:30 PM", "tomorrow at 9")
- Deterministic fast-path pattern matching (all supported patterns)
- UUIDv7 ID generation (format validation, sortability, uniqueness)
- Write debouncing behavior (buffering, flush timing, graceful shutdown)
- Tool execution with various parameters
- File lifecycle tests: create/update/delete timer file, alarm stop-all behavior, corrupted JSON quarantine

### Integration Tests
- STT → Tool call → TTS flow
- Multiple concurrent timers
- State persistence and recovery
- Audio alert generation and playback

### End-to-End Tests
- Voice command recording + processing + response
- Fast-path latency validation (<200ms for obvious commands)
- LLM fallback latency validation (<500ms for ambiguous commands)
- Latency measurements with different LLM providers
- Edge cases (cancelling non-existent timers, invalid times)
- Restart tests: set timers/alarms, restart orchestrator, verify rehydration and expected alert behavior
- Debounce validation: rapid alarm state changes during active ringing

### Performance Tests
- Memory usage with 100+ active timers
- CPU usage during monitoring loop
- File scan/listing performance in `timers/` directory
- Burst behavior: 100 simultaneous timers with atomic write contention

---

## Key Files to Create/Modify

### New Files
```
orchestrator/tools/
  ├── __init__.py
  ├── timer.py              # Timer implementation
  ├── alarm.py              # Alarm implementation
  ├── state.py              # Persistent state store (UUIDv7, debouncing)
  ├── router.py             # Tool execution router
  ├── monitor.py            # Background monitoring
  ├── parser.py             # Deterministic fast-path + time expression parsing
  └── uuid_utils.py         # UUIDv7 generation utilities

orchestrator/
  └── alerts.py             # Audio alert generation
```

### Modified Files
```
orchestrator/gateway/quick_answer.py     # Tool call support
orchestrator/main.py                     # Tool router integration
orchestrator/config.py                   # Tool-related settings
```

### Documentation
```
TOOL_CALLS_AND_TIMER_PLAN.md     # This document
TOOL_IMPLEMENTATION_EXAMPLES.md   # Code walkthroughs
TOOL_DEVELOPMENT_GUIDE.md         # Adding new tools
```

---

## Success Criteria

### MVP (Minimum Viable Product)
- ✓ User can set a timer via voice
- ✓ Timer expires and announces via TTS
- ✓ User can cancel timer
- ✓ <200ms latency for obvious commands (fast-path)
- ✓ <500ms latency for ambiguous commands (LLM fallback)
- ✓ State persists as per-item files in workspace `timers/`
- ✓ UUIDv7 used for all timer/alarm identifiers
- ✓ Write debouncing implemented for alarm state updates
- ✓ Timers/alarms survive full orchestrator restart
- ✓ Timer files are deleted when timers complete
- ✓ User notified of recovered state on startup

### Extended
- ✓ Set/cancel/query alarms
- ✓ Stop actively ringing alarms via voice command
- ✓ "Stop alarm" with no target stops all ringing alarms
- ✓ Auto-stop alarm on cut-in (speech detected)
- ✓ Support snooze functionality (pause and resume)
- ✓ Query active timers/alarms via voice
- ✓ Time expression parsing (various formats)
- ✓ Multiple concurrent timers/alarms
- ✓ Alarms keep ringing until explicitly stopped or dismissed
- ✓ Alerts triggered immediately if they expired during downtime
- ✓ Optional event log retention (30 days) without retaining completed timer files

### Production-Ready
- ✓ Comprehensive error handling
- ✓ Performance optimized
- ✓ Full logging and monitoring
- ✓ User documentation
- ✓ Easy framework for adding new tools
- ✓ Timer/alarm file validation on startup
- ✓ Graceful shutdown (flush state to disk)
- ✓ Backward-compatible file schema migration strategy (`schema_version`)

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| LLM doesn't recognize tool calls | High | Medium | Clear system prompt + few-shot examples |
| Latency exceeds 500ms | High | Medium | Async execution + caching |
| State lost on gateway restart | High | Low | File-per-item persistence in `timers/` + startup rehydration |
| File corruption during write | High | Low | Temp file + atomic rename + quarantine invalid files |
| Clock skew / timezone errors | Medium | Medium | Store UTC epoch only; convert for speech at render time |
| Audio conflicts (alert + speech) | Medium | Medium | Queue management + priority levels |
| Time parsing ambiguity | Medium | High | Ask clarification for ambiguous inputs |
| Timer expires during downtime | Medium | High | Load expires_at, trigger alert immediately if past threshold on restart |
| Alarm missed during downtime | Medium | High | Store trigger_time, fire alert on startup if within recent window (1 hour) |

---

## Notes

- Start with timer skill as proof-of-concept
- Use OpenAI function calling format for LLM compatibility
- Design tool interface to be easily extensible
- All tools should execute synchronously or fast enough for voice responsiveness
- Consider accessibility (audio-only interface has no visual feedback)

