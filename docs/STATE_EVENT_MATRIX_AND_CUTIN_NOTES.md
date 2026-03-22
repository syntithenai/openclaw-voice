# OpenClaw Voice: State/Event Matrix, Cut-In Pathways, and Transcription Audio Guardrails

_Date: 2026-03-16_

This is a compact behavior map for the **Python orchestrator** (`openclaw-voice/orchestrator/main.py`) with emphasis on:

1. Play-button and wake/sleep triggers
2. Cut-in pathways
3. How audio is selected and filtered before transcription to minimize TTS echo transcripts while preserving user speech

---

## 1) Compact state/event matrix

### Legend
- **WS** = wake state (`AWAKE` / `ASLEEP`)
- **VS** = voice state (`IDLE` / `LISTENING` / `SENDING`)
- **Cue** = wake click or sleep swoosh/exhale/sigh variants
- **MPD** = music manager state interactions

### Trigger matrix (behavior by event source)

| Event source | If alarms ringing | If WS=ASLEEP | If WS=AWAKE | TTS impact | Music impact | Cue impact |
|---|---|---|---|---|---|---|
| Hardware `play_pause` / `play` / `pause` | **Consumes event** to stop ringing alarms | `trigger_wake("play button")` | `trigger_sleep("play button")` (when wakeword enabled) | Wake/sleep path sets `tts_stop_event`; sleep clears queued TTS | Wake: pause-if-playing async; Sleep: optional restore of paused-for-wake music | Wake cue on wake; sleep cue on sleep |
| Hardware `next` / `previous` | **Consumes event** to stop ringing alarms | Same as play toggle | Same as play toggle | Same as play toggle | Same as play toggle | Same as play toggle |
| Hardware `play_pause_long` | N/A | Forces wake | Forces wake (idempotent wake behavior) | Stops TTS via wake path stop event | Pause-if-playing async | Wake cue |
| Hardware `phone` | N/A | Forces wake | Forces wake | Stops TTS via wake path stop event | Pause-if-playing async | Wake cue |
| Web UI `mic_toggle` | No alarm shortcut in this path | Enables mic + sets awake/listening | Toggles to sleep when awake; toggles back awake from asleep | `tts_stop_event` is not always applied exactly like media-key path | Pauses music when waking from UI path | Hotword indicator updated; cues are less centralized in this UI path |
| Wakeword detector (sleep loop) | N/A | Detect wake, wake/listen | N/A | Does not directly clear queue; transitions into normal listen pipeline | Pauses music if playing and marks paused-for-wake | Wake cue |
| Wake timeout | N/A | N/A | Transitions to asleep if idle/no TTS/no pending transcripts | Keeps awake while TTS is active or queued | If music still playing and configured, remains aligned with music sleep logic | Sleep cue (timeout variant) |

---

## 2) Lifecycle-stage implications (compact)

| Lifecycle stage | Play-like trigger outcome | Wakeword outcome | Net risk/benefit |
|---|---|---|---|
| Startup/init (handlers not fully started) | May do nothing until detector/web handlers active | May do nothing until wake detector initialized | Safe but can feel “dead” briefly |
| ASLEEP + idle | Wakes and starts listening | Wakes and starts listening | Consistent user intent path |
| AWAKE + listening | Play toggle sleeps system | N/A | Fast explicit sleep; can interrupt pending interaction |
| During TTS playback | Wake/sleep can interrupt playback; cut-in can force interruption | Wakeword not primary here unless asleep branch active | Responsive interruption, but requires guardrails vs echo loops |
| During ringing alarms | Play/next/prev dismiss alarm first | N/A | Prevents accidental wake/sleep when user intent is alarm dismissal |
| During music playback (`MUSIC_SLEEP_DURING_PLAYBACK=true`) | Wake/play paths can pause music and wake temporarily | Wakeword can pause/duck music then wake | Strong anti-false-transcript posture while music active |

---

## 3) Cut-in pathways

There are **three distinct cut-in-like pathways**:

### A) TTS cut-in (user speaks over assistant speech)

Primary path while `tts_playing=True`.

Gate logic combines:
- Minimum elapsed playback (`VAD_CUT_IN_MIN_MS`)
- VAD speech signal (WebRTC/Silero path)
- RMS checks:
	- `rms_cutin` (processed frame)
	- `rms_excess = max(0, rms_raw - tts_rms_baseline)`
- Consecutive frame count (`VAD_CUT_IN_FRAMES`)
- Optional Silero confidence gate (`VAD_CUT_IN_USE_SILERO`, `VAD_CUT_IN_SILERO_CONFIDENCE`)

On trigger:
1. Set `tts_stop_event`
2. Build/continue cut-in chunk with small pre-roll (`CUT_IN_PRE_ROLL_MS`)
3. Set/refresh awake activity timestamps
4. Activate **TTS hold window** (`VAD_CUT_IN_TTS_HOLD_TIMEOUT_MS`) and clear queued TTS
5. Optionally pause music if playing

Goal: interrupt quickly, keep user’s interruption audio, and avoid immediate TTS re-entry loops.

### B) Alarm cut-in (user speaks while alarm ringing)

Separate path when alarms are ringing and TTS isn’t dominating:
- Uses VAD + RMS thresholds + frame hit counting
- On trigger: stops ringing alarms immediately, resets chunk state, applies brief audio drop window to suppress alarm residue transcript artifacts.

Goal: prioritize alarm dismissal by voice while preventing bell noise from becoming transcript text.

### C) Music-sleep cut-in duck (hotword assist while asleep during music)

When asleep due to active music playback:
- Voice candidate can trigger temporary MPD ducking (`MUSIC_CUT_IN_DUCK_RATIO` + timeout)
- This creates a cleaner window for wakeword detection
- If wakeword is detected, ducking is restored then normal wake flow pauses music.

Goal: increase wakeword reliability during loud playback without permanently changing volume.

---

## 4) How transcription input is managed to limit TTS echo while preserving speech

This is a layered strategy; no single mechanism does all the work.

### Layer 1: Audio source authority and arbitration

Runtime source selection (`WEB_UI_AUDIO_AUTHORITY`):
- `native`: local mic only
- `browser`: browser PCM when connected+fresh, fallback local mic otherwise
- `hybrid`: same handoff behavior with explicit browser participation semantics

This reduces source ambiguity and keeps one authoritative capture stream at a time in practice.

### Layer 2: Pre-roll ring buffer strategy

- Continuous ring buffer preserves context before VAD trigger
- On wake:
	- Optional full clear (`WAKE_CLEAR_RING_BUFFER=true`) to avoid stale/ghost audio
	- Otherwise uses reduced pre-roll window to avoid including hotword itself excessively
- On cut-in:
	- Uses **small** pre-roll (`CUT_IN_PRE_ROLL_MS`) for responsiveness

This balances “don’t lose first syllables” vs “don’t over-capture stale/noisy content.”

### Layer 3: During TTS, normal STT path is suppressed

While TTS is playing:
- Normal speech→chunk→transcribe path is intentionally suppressed
- Only the cut-in path can open a chunk for transcription

This is one of the strongest anti-echo protections: if no user interruption, no transcript should be produced from ongoing assistant audio.

### Layer 4: Echo-aware gating for cut-in detection

- Tracks `tts_rms_baseline` during playback
- Uses `rms_excess` above baseline and VAD conditions
- Optional Silero confidence gate for additional robustness

This tries to differentiate user speech from speaker bleed.

### Layer 5: AEC + raw-frame preservation tradeoff

- AEC is applied for many real-time decisions, but ring buffer stores raw frames to avoid over-aggressive speech loss
- Code comments explicitly note that over-aggressive AEC can remove user speech with echo

Net effect: tries to retain user intelligibility even if that means relying on downstream transcript filtering for residual echoes.

### Layer 6: Transcript normalization + low-signal filtering

After STT returns text, pipeline filters:
- Blank/marker outputs (`[inaudible]`, punctuation-only, descriptor-only)
- Known low-signal artifacts (“thank you”, “sigh”, etc. list)
- Self-echo heuristics using overlap with recent/current TTS text (`is_likely_tts_self_echo`)

This is a textual final defense against TTS self-capture loops.

### Layer 7: TTS queue dedupe and stale request dropping

- Drops stale reply items older than current request
- Dedupes near-duplicate TTS within time window
- Holds/restricts TTS restart after cut-in

This prevents transcript artifacts from immediately re-amplifying into new spoken loops.

---

## 5) Why this generally avoids “transcribe my own TTS” while minimizing lost user audio

The system is effectively doing:

1. **Prevent**: suppress normal STT ingestion during TTS
2. **Permit only explicit interruption**: cut-in path must satisfy stronger gates
3. **Preserve user onset**: keep small pre-roll and avoid over-trusting AEC alone
4. **Clean up residuals**: transcript post-filters and self-echo checks
5. **Avoid re-looping**: TTS hold + queue hygiene

That combination is why it can usually reject assistant echo while still capturing genuine user interruption with minimal clipping.

---

## 6) Practical parity note (hardware vs web vs wakeword)

Behavior is close in effect, but implementation is still multi-path:
- Hardware media-key path has nested wake/sleep helpers with rich side effects
- Web `mic_toggle` path mirrors behavior but is partially separate logic
- Wakeword path in main loop has its own wake sequence path

So outcomes are mostly aligned, but strict “single function for all triggers” parity is not fully centralized yet.

