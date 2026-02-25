# STT Quality Gate Plan

## Goal
Minimize gateway calls by filtering non-meaningful Whisper transcripts while avoiding loss of legitimate speech. The system must never hard-drop user speech; it should defer, merge, and re-score before deciding to send upstream.

## Non-Goals
- Do not modify Whisper models or decoding parameters in this plan.
- Do not rely on brittle, language-specific filters only.
- Do not block transcripts with hard rejects except for explicit safety rules (see Exceptions).

## Key Idea
Use a soft, score-based gate that defers low-quality segments, merges them with upcoming segments, and only sends when the combined transcript passes a quality threshold or a max wait window expires.

## Signals (Score Inputs)
Each signal is weak on its own but strong in combination. Use weighted scoring.

### 1) ASR Proxy Signals (from Whisper)
Whisper does not expose a single confidence score, but these are commonly available:
- `avg_logprob`: higher is better
- `no_speech_prob`: higher implies noise or silence
- `compression_ratio`: high ratio suggests repetitive or low-quality output

### 2) Lexical Quality
- Dictionary hit rate: proportion of tokens found in a basic wordlist
- Vowel ratio: low vowel ratio can indicate non-words ("tch", "mm")
- Token length stats: excessive 1-2 character tokens reduce score

### 3) Repetition / Patterning
- Unique token ratio: low uniqueness indicates repetition
- N-gram repetition rate: repeated 1-3-gram patterns reduce score
- Sequence entropy: low entropy means repetitive content

### 4) Numeric Dominance
- Numeric token ratio: if most tokens are numbers, reduce score
- Exceptions: allow if at least one anchor word is present ("percent", "battery", "temperature", "speed")

### 5) Punctuation and Noise Markers
- Excessive punctuation or non-alphanumeric characters reduce score

## Gating Decision Policy
Use a tiered decision:

1) Score >= PASS threshold
- Send immediately.

2) Score between HOLD_LOW and PASS
- Defer. Buffer the transcript and wait for the next segment.
- Merge with the next segment and re-score.

3) Score below HOLD_LOW
- Defer and wait up to MAX_WAIT_MS or MAX_SEGMENTS.
- If still below threshold after max wait, send with a "low_quality" flag.

## Exceptions (Never Send)
Only enforce if required by policy; otherwise handle via defer-and-merge:
- Empty or whitespace-only transcript
- Pure punctuation (no alphanumerics)

## Example Classifications
- "Tch!" -> low vowel ratio + short token + no word hits -> defer
- "1,2,1,1,1..." -> repeated n-grams + numeric dominance -> defer
- "1.5% 1.5% 1.5%" -> numeric dominance + repetition -> defer

## Parameters (Initial Defaults)
- PASS threshold: 0.65
- HOLD_LOW: 0.40
- MAX_WAIT_MS: 1200
- MAX_SEGMENTS: 2
- Dictionary hit minimum: 0.25
- Unique token ratio minimum: 0.35
- Numeric token ratio maximum (without anchor word): 0.60

These should be tuned with real logs.

## Scoring Outline (Pseudo)
- Start score at 1.0
- Apply penalties for:
  - High `no_speech_prob`
  - Low `avg_logprob`
  - High `compression_ratio`
  - Low dictionary hit rate
  - Low unique token ratio
  - High repetition rate
  - High numeric token ratio without anchors
  - Low vowel ratio

Clamp score to [0, 1].

## Instrumentation
Log a compact score summary for each segment:
- transcript
- score
- component penalties
- decision (pass/hold/low_quality)
- wait and merge stats

## Testing Plan
1) Unit tests
- Tokenization and scoring functions
- Repetition detection
- Numeric dominance logic

2) Offline replay tests
- Run recorded STT logs through the scorer
- Compare pass/hold rates with manual labeling

3) Live smoke test
- Enable in shadow mode (no gating) and compare decisions
- Enable with gating in a small environment

## Rollout Steps
1) Implement scoring with logging only
2) Enable defer-and-merge in shadow mode
3) Turn on gating with conservative thresholds
4) Tune thresholds based on observed false holds

## Open Questions
- Which Whisper runtime is used (OpenAI, faster-whisper, whisper.cpp) and which proxy fields are available?
- Should we tag low-quality sends differently for downstream handling?
- Do we need per-language dictionaries or a lightweight wordlist?
