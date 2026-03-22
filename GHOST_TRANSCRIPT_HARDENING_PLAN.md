# Ghost Transcript Hardening Plan

## Background

Whisper hallucinates coherent text during silence. Its decoder is a language model trained on
680K hours of YouTube audio. When no speech is present it generates the most probable completion
from its training distribution — YouTube outros ("Thanks for watching!"), subtitle watermarks
("Subtitles by the Amara.org community"), and infinite repetition loops where the decoder gets
stuck on a high-probability token.

The Vexa production team collected 135 reproducible English hallucination phrases after thousands
of hours of real production audio. Their mitigation layers (VAD pre-gating, `condition_on_previous_text=False`,
exact-string blocklist, repeated-output detection, `beam_size=1`) address the problem at multiple
independent levels. OpenClaw Voice already has a runtime ghost-transcript filter (`decide_ghost_transcript`
in `orchestrator/main.py`) but currently has no mitigations at the Whisper backend level. Gaps in
`GHOST_ARTIFACT_TOKENS` also mean many known Vexa hallucinations pass unfiltered from the backend
before the orchestrator even sees them.

---

## Current State

### Existing defences (orchestrator level)
- `decide_ghost_transcript()` — context-aware scoring gate that rejects transcripts that look like
  acoustically-plausible but semantically implausible echoes or artefacts.
- `GHOST_ARTIFACT_TOKENS` — small exact-match set (~15 entries: "hello", "hi", "thanks",
  "thank you", "hmm", "sigh", "we'll be right back", …).
- `GHOST_ARTIFACT_PATTERNS` — one regex pattern ("I'm going to go ahead and stop the recording").
- Self-echo similarity scoring — suppresses transcripts that closely match recent TTS output.

### Missing defences (backend / pre-gate level)
| Gap | Impact |
|---|---|
| `docker/whisper/app.py` calls `MODEL.transcribe(path)` with no extra options | `condition_on_previous_text` defaults to `True`; hallucination feedback loops are live |
| `docker/whisper/app.py` uses default `beam_size` (5) | Longer hallucination loops because beam search finds more plausible completions |
| `docker/whisper/app.py` does not check `no_speech_prob` per segment | Segments with very high no-speech probability pass through silently |
| `docker/whisper/app_whispercpp.py` passes no beam/greedy flags to whisper-cli | Same beam-search risk on the CPP backend |
| `openclaw/docker/whisper/app.py` (openclaw proper) also uses bare `transcribe()` | Same issues |
| No hallucination blocklist file | ~120+ known production phrases not in `GHOST_ARTIFACT_TOKENS` |
| VAD is `webrtc` by default; `VAD_CUT_IN_USE_SILERO=false` by default | Silero (trained specifically for voice-activity) is unused as a pre-gate |
| No repeated-output detection at backend or orchestrator | Looping hallucinations ("Thank you… thank you… thank you…") pass word-count gate |

---

## Proposed Changes

### 1. Whisper backend: `condition_on_previous_text=False`

**File:** `docker/whisper/app.py`

Change the `MODEL.transcribe()` call to pass `condition_on_previous_text=False`. This prevents a
hallucinated text in one 30-second window from seeding the next window's prompt, which is the primary
cause of runaway repetition loops.

```python
# Before
result = MODEL.transcribe(temp_audio.name)

# After
result = MODEL.transcribe(
    temp_audio.name,
    condition_on_previous_text=False,
    beam_size=1,
)
```

Expose these as env vars (`WHISPER_CONDITION_ON_PREVIOUS_TEXT=false`, `WHISPER_BEAM_SIZE=1`) so
they can be tuned per deployment without a rebuild.

### 2. Whisper backend: per-segment `no_speech_prob` filter

**File:** `docker/whisper/app.py`

After transcribing, drop any segment where `no_speech_prob` exceeds a threshold (configurable,
default 0.6). Return an empty text if all segments are dropped. This is a cheap second gate even
though OpenAI's own docs note the metric is imprecise.

```python
NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))

def _normalize_segments(raw_segments, no_speech_threshold: float = 0.6):
    normalized = []
    for row in raw_segments or []:
        if float(row.get("no_speech_prob", 0.0)) >= no_speech_threshold:
            continue
        ...
    return normalized
```

### 3. whisper.cpp backend: greedy decode flag

**File:** `docker/whisper/app_whispercpp.py`

Add `--best-of 1` (greedy decode) to the `whisper-cli` command. The whisper.cpp CLI equivalent of
`beam_size=1` is `--best-of 1`; also add `--no-context` which corresponds to
`condition_on_previous_text=False`.

```python
cmd = [
    WHISPER_CPP_PATH,
    "-m", MODEL_PATH,
    "-f", temp_audio_path,
    "-oj",
    "-of", temp_audio_path,
    "--best-of", "1",    # greedy decode, no beam search
    "--no-context",      # do not condition on prior output
]
```

Expose as env vars `WHISPER_CPP_BEST_OF` and `WHISPER_CPP_NO_CONTEXT`.

### 4. Hallucination blocklist file

**New file:** `docker/whisper/hallucinations/en.txt`

Seed this with the full Vexa English blocklist (135 phrases from
`https://raw.githubusercontent.com/Vexa-ai/vexa/main/services/WhisperLive/hallucinations/en.txt`).
The file is one phrase per line (case-insensitive exact match, allowing leading/trailing whitespace).

Load the blocklist in the transcribe endpoint and strip any result that is an exact (case-insensitive,
stripped) match against a blocklist entry, returning `""` instead. This is language-independent
infrastructure: additional `es.txt`, `pt.txt`, `ru.txt` files can be dropped in and picked up
without code changes.

### 5. Repeated-output detection at the backend

**File:** `docker/whisper/app.py`

After collecting all segment texts, detect if the same short phrase is repeated many times.
Heuristic: if the same sequence of ≤8 words appears more than 8 times, truncate to one
occurrence and log a warning. This catches the "Thank you Mr. President" infinite loop class.

### 6. Expand `GHOST_ARTIFACT_TOKENS` in the orchestrator

**File:** `orchestrator/main.py`

Add the subset of Vexa phrases that are single short sentences plausibly entering the voice
assistant context and are never real commands — e.g.:

```
"thanks for watching"
"thank you for watching"
"i'll see you next time"
"see you next time"
"thank you for your time"
"thank you so much for joining us"
"subtitles by the amara org community"
"god bless you"
"we ll be right back"        # already present
"well be right back"
"uh huh"
"bye"
"bye bye"
```

Multi-line / paragraph-length hallucinations don't need to be added here because they will be
caught first by the blocklist at the backend level (change 4) and the repeated-output detector
(change 5).

### 7. Repeated-output detection at the orchestrator

**File:** `orchestrator/main.py`

Add a check inside `decide_ghost_transcript` (or just before the call site): if the same **phrase
core** has been seen in the last N transcripts (configurable, default 8) and the time gap between
consecutive occurrences is very short (< 5 s), treat it as a stuck-loop hallucination and reject.

This is orthogonal to self-echo similarity: self-echo compares to recent TTS; this compares to
recent *input* transcripts.

### 8. Post-collection Silero confidence gate (pre-transcription)

**Note:** Silero is already fully integrated — `VAD_TYPE=silero` makes it the primary utterance
VAD, and `VAD_CUT_IN_USE_SILERO=true` (already set in `.env`) gates cut-in detection. These are
not disabled features.

The gap that remains: once a speech chunk has been collected and VAD has fired, the audio is sent
to Whisper unconditionally. A collected chunk can still be mostly silence if the user paused
mid-utterance or if WebRTC VAD fired on a transient noise. Adding a **post-collection Silero
confidence re-check** in the orchestrator's transcription path — computing the average Silero
confidence over the collected frames and skipping the Whisper call if it falls below a threshold
(e.g. 0.3) — provides an extra upstream gate before bytes ever leave for the Whisper service.

This is orthogonal to `VAD_TYPE` and does not require disabling or replacing anything already
working. The Silero model is already loaded when `VAD_CUT_IN_USE_SILERO=true`; its instance can
be reused for this check at zero additional memory cost.

---

## Priority Order

| Priority | Change | Risk | Effort |
|---|---|---|---|
| 1 | `condition_on_previous_text=False` + `beam_size=1` (backends) | Very low — only changes hallucination risk | Small |
| 2 | Hallucination blocklist file (en.txt) at backend | Very low | Small |
| 3 | Expand `GHOST_ARTIFACT_TOKENS` | Very low | Trivial |
| 4 | Per-segment `no_speech_prob` filter | Low — might drop legitimate short utterances if threshold too tight | Small |
| 5 | Repeated-output detection (backend) | Low | Small |
| 6 | Repeated-output detection (orchestrator) | Medium — must not suppress legitimate rapid commands | Medium |
| 7 | whisper.cpp `--best-of 1 --no-context` | Very low | Trivial |
| 8 | Post-collection Silero confidence re-check before Whisper call | Low — reuses already-loaded model; needs threshold tuning | Small |

---

## What NOT to Do

- Do not set `no_speech_prob` threshold below 0.5 — too many legitimate soft-spoken transcripts
  will be dropped.
- Do not add very short common words ("okay", "yes", "no") to the backend blocklist — these are
  legitimate voice commands. They are already gated downstream by `GHOST_ARTIFACT_TOKENS` and the
  `require_question_for_acks` logic.
- Do not add `GHOST_ARTIFACT_PATTERNS` regex for every Vexa phrase — the per-line exact blocklist
  file (change 4) is easier to maintain and faster.

---

## Files Affected

| File | Change |
|---|---|
| `docker/whisper/app.py` | `condition_on_previous_text=False`, `beam_size=1`, `no_speech_prob` per-segment filter, repeated-output detection, blocklist loading |
| `docker/whisper/app_whispercpp.py` | `--best-of 1 --no-context` CLI flags |
| `docker/whisper/hallucinations/en.txt` | New: seeded from Vexa blocklist |
| `orchestrator/main.py` | Expand `GHOST_ARTIFACT_TOKENS`, add repeated-input loop detection |
| `.env.example` | Document new env vars; consider enabling Silero by default |
