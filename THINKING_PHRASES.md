# Thinking Phrases Feature

## Overview

When the Quick Answer system determines it needs to escalate to the upstream gateway (for complex queries that require the full conversational agent), it now plays a brief "thinking" phrase to provide audio feedback to the user that the system is processing their request.

## Implementation Details

### Thinking Phrases Collection

A set of 10 natural-sounding phrases that convey the system is processing:

```python
THINKING_PHRASES = [
    "thinking",
    "just a sec", 
    "onto it",
    "ok",
    "let me think",
    "one moment",
    "hmm",
    "let me check",
    "give me a sec",
    "working on it",
]
```

### When Phrases are Played

1. **Normal Escalation**: When Quick Answer LLM returns `USE_UPSTREAM_AGENT`
2. **Error Fallback**: When Quick Answer fails with an exception

### Selection Algorithm

Uses `random.choice()` to randomly select one phrase from the collection each time escalation occurs. This provides variety and makes the system feel more natural.

### Flow Diagram

#### Before (No Thinking Phrase)
```
User: "What's the weather tomorrow?"
  ↓
Quick Answer: [checks] → USE_UPSTREAM_AGENT
  ↓
[silence while gateway processes]
  ↓
Gateway: "Tomorrow will be sunny..."
```

#### After (With Thinking Phrase)
```
User: "What's the weather tomorrow?"
  ↓
Quick Answer: [checks] → USE_UPSTREAM_AGENT
  ↓
TTS: "just a sec" (random selection)
  ↓
Gateway: "Tomorrow will be sunny..."
```

## User Benefits

1. **Audio Feedback**: User knows the system heard them and is processing
2. **Natural Feel**: Random selection prevents repetitive responses
3. **Reduced Perceived Latency**: Something happens immediately even if full answer takes longer
4. **Error Transparency**: Even on errors, user gets acknowledgment

## Technical Implementation

### Files Modified

1. **orchestrator/gateway/quick_answer.py**
   - Added `THINKING_PHRASES` list
   - Added `get_random_thinking_phrase()` function

2. **orchestrator/main.py**
   - Import `get_random_thinking_phrase` when Quick Answer enabled
   - Play thinking phrase on escalation (line ~1241)
   - Play thinking phrase on error fallback (line ~1258)

### Code Locations

**Escalation (Normal Flow)**:
```python
else:
    # Need to escalate to gateway
    logger.info("← QUICK ANSWER: Escalating to upstream")
    
    # Play a thinking phrase while gateway processes
    thinking_phrase = get_random_thinking_phrase()
    await submit_tts(thinking_phrase, request_id=current_request_id)
```

**Error Fallback**:
```python
except Exception as exc:
    logger.error("Quick answer failed: falling back to gateway")
    
    # Play a thinking phrase on error too
    try:
        thinking_phrase = get_random_thinking_phrase()
        await submit_tts(thinking_phrase, request_id=current_request_id)
    except Exception as tts_exc:
        logger.error("Failed to play thinking phrase: %s", tts_exc)
```

## Configuration

No configuration needed - feature is automatically enabled when `QUICK_ANSWER_ENABLED=true`.

## Testing

To test different thinking phrases, you can:

1. Enable Quick Answer in `.env`:
   ```bash
   QUICK_ANSWER_ENABLED=true
   QUICK_ANSWER_LLM_URL=http://localhost:8080/v1/chat/completions
   ```

2. Ask questions that require gateway escalation:
   - "What's the weather tomorrow?" (needs current data)
   - "Tell me a story" (complex, creative task)
   - "What happened in the news today?" (time-sensitive)

3. Listen for the thinking phrase before the full answer

4. Try multiple times to hear different random selections

## Future Enhancements

Potential improvements:
- User-configurable phrase list via config
- Different phrase sets for different contexts (error vs. normal)
- Tone/emotion markers for TTS emphasis
- Configurable probability (sometimes silent, sometimes phrase)
