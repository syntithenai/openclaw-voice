# Tool System - Timers & Alarms

## Overview

The voice orchestrator now includes a built-in tool system for timers and alarms that enables:
- **Immediate responses** without gateway round-trips
- **Deterministic fast-path** parsing for obvious commands (<200ms)
- **LLM fallback** with OpenAI function calling for complex requests
- **Persistent state** with file-based storage
- **Background monitoring** for expiration detection
- **Audio alerts** with bell sounds

## Features

### Timers
- Set countdown timers with natural language ("5 minutes", "2 hours")
- Optional naming ("pizza timer", "workout")
- Single bell sound on completion
- Automatic deletion after expiration

### Alarms
- Set alarms for specific times ("6:30 AM", "tomorrow 9am")
- Optional naming ("wake up", "meeting")
- Continuous bell ringing until stopped
- Resume ringing state after restart (1-hour missed alarm window)

### Commands

#### Timer Commands
```
"set a timer for 5 minutes"
"set timer 30 seconds"
"set a 10 minute timer called pizza"
"cancel pizza timer"
"cancel all timers"
"list timers"
"how many timers"
```

#### Alarm Commands
```
"set alarm for 6:30 AM"
"set alarm tomorrow at 9am"
"wake me up at 7"
"stop alarm"
"stop the wake up alarm"
"cancel morning alarm"
"list alarms"
```

## Configuration

Add to `.env`:

```bash
# Tool System
TOOLS_ENABLED=true
TOOLS_PERSIST_DIR=timers            # Directory for persistence (relative to workspace)
TOOLS_DEBOUNCE_MS=75                # Write debouncing window
TOOLS_MONITOR_INTERVAL_MS=100       # Expiration check interval

# Quick Answer LLM (required for tool calling)
QUICK_ANSWER_ENABLED=true
QUICK_ANSWER_LLM_URL=http://localhost:8080/v1/chat/completions
QUICK_ANSWER_MODEL=qwen2.5:3b
QUICK_ANSWER_TIMEOUT_MS=5000
```

## Architecture

### Fast-Path Parsing
Regex-based pattern matching bypasses LLM for obvious commands:
- Target latency: <200ms for simple commands
- ~90% of timer/alarm commands hit fast-path
- Graceful fallback to LLM for complex requests

### Persistence
- One JSON file per timer/alarm in `<workspace>/timers/active/`
- Atomic writes with temp+rename
- Quarantine for corrupted files
- Optional audit log in `events/`

### File Format
```json
{
  "id": "timer-01h2xcejqtf2nbrexx3vqjhp41",
  "name": "pizza",
  "created_at": 1704067200.123,
  "expires_at": 1704067500.456
}
```

### UUIDv7 Identifiers
- Timestamp-ordered 128-bit UUIDs
- Format: `timer-01h2xcejqtf2nbrexx3vqjhp41`
- Collision-safe for distributed systems
- Sortable by creation time

### Write Debouncing
- Non-critical alarm state updates buffered for 50-100ms
- Reduces I/O by ~90% during alarm ringing
- Immediate flush on shutdown

## Tool Call Format

OpenAI function calling format:

```json
{
  "tool_calls": [
    {
      "function": {
        "name": "set_timer",
        "arguments": {
          "duration_seconds": 300,
          "name": "pizza"
        }
      }
    }
  ]
}
```

## Background Monitoring

ToolMonitor runs asyncio loop checking every 100ms:
- Timer expiration → play bell + TTS announcement
- Alarm triggering → TTS announcement + start ringing
- Alarm ringing → bell sound every 3 seconds

## Testing

Run the test suite:

```bash
./test_tool_system.py
```

Tests cover:
- Fast-path parser patterns
- Timer lifecycle (create, list, expire)
- Alarm creation and cancellation
- Alert sound generation
- Monitor expiration detection

## Extending the System

### Adding New Tools

1. **Define tool in `TOOL_DEFINITIONS`** (quick_answer.py):
```python
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform arithmetic calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
}
```

2. **Add method to ToolRouter** (router.py):
```python
async def calculate(self, expression: str) -> str:
    try:
        result = eval(expression)  # Use safe eval in production
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

3. **Register in execute_tool** (router.py):
```python
elif tool_name == "calculate":
    return await self.calculate(**kwargs)
```

4. **Add fast-path patterns** (parser.py) [optional]:
```python
CALC_PATTERNS = [
    r"(?:calculate|compute|what'?s)\s+(.+)",
]
```

## Best Practices

1. **Fast-path patterns** - Add regex for common phrasings
2. **Natural time parsing** - Support flexible time expressions
3. **Atomic writes** - Use temp file + rename for data integrity
4. **Graceful degradation** - Fall back to upstream on errors
5. **Idempotent operations** - Safe to retry or duplicate
6. **Deterministic responses** - Same input → same output

## Troubleshooting

### Timers not persisting
- Check `TOOLS_PERSIST_DIR` exists and is writable
- Look for `.quarantine` files indicating corruption

### Fast-path not matching
- Check logs for "Fast-path tool execution"
- Add pattern to parser.py and test with test_tool_system.py

### Alarms not ringing after restart
- Ensure `load_from_disk()` called at startup
- Check alarm times are within 1-hour missed window

### Tool calls not working
- Verify LLM supports OpenAI function calling
- Check logs for "LLM requested tool call"
- Ensure `TOOLS_ENABLED=true` and `QUICK_ANSWER_ENABLED=true`

## Performance

- **Fast-path latency**: 50-200ms (regex parsing + execution)
- **LLM fallback latency**: 300-800ms (depends on model)
- **File I/O**: <5ms per operation (SSD)
- **Monitor overhead**: <1% CPU (100ms check interval)
- **Memory**: ~100KB per 1000 active timers/alarms

## Limitations

- Time zone support limited to system timezone
- No recurring alarms (future enhancement)
- Bell sounds cannot be customized (hardcoded)
- No snooze functionality (future enhancement)
- Maximum 1-hour missed alarm window

## Future Enhancements

- [ ] Recurring timers/alarms (daily, weekly)
- [ ] Snooze functionality (5/10/15 minute options)
- [ ] Custom alert sounds (MP3/OGG support)
- [ ] Time zone awareness
- [ ] Calendar integration
- [ ] Group timer management
- [ ] Timer pause/resume
- [ ] Alarm escalation (increasing volume)
