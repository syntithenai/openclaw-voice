# Music Integration and Media Keys - Complete Guide

## Overview

The orchestrator now has deep integration with MPD music playback and hardware media controls. This includes:

1. **Automatic sleep during music playback** - Prevents voice transcription while music is playing
2. **Wake-on-voice** - Stop music instantly when wake word or voice is detected
3. **Auto-resume** - Restart music after 5 seconds of silence
4. **Hardware button controls** - Full conference speaker button integration

## Key Features

### 1. Music Playback State Management

**When music starts playing:**
- Orchestrator automatically goes to sleep (if `MUSIC_SLEEP_DURING_PLAYBACK=true`)
- Prevents false transcriptions from music
- System remains responsive to wake word

**When wake word detected:**
- Music stops immediately
- Wake notification sound plays (if enabled)
- System starts listening for voice commands
- Music auto-resume timer starts

**When voice cut-in detected (speaking over TTS):**
- TTS playback stops immediately
- Music stops immediately  
- System captures voice input
- Music auto-resume timer starts

**Auto-resume behavior:**
- If no voice command detected for 5 seconds (configurable)
- Music automatically resumes playing
- System returns to sleep (if music sleep enabled)

### 2. Conference Speaker Button Functions

#### Mute Button
- **Function**: Toggle microphone mute/unmute
- **Behavior**: 
  - Press once to mute microphone (silent input)
  - Press again to unmute
  - Mute state persists until toggled
  - Does NOT affect MPD volume

#### Play Button
- **Function**: Smart play/stop
- **Behavior**:
  - If queue is empty: Add 50 random tracks and start playing
  - If music is playing: Pause
  - If music is paused: Resume
  - Number of random tracks configurable via `MUSIC_RANDOM_TRACK_COUNT`

#### Phone Button
- **Function**: Trigger wake word sequence
- **Behavior** (same as wake word):
  1. Stop any music/TTS playback
  2. Play wake notification sound
  3. Unmute microphone (if muted)
  4. Wake system and start listening
  5. Capture voice input
  6. Auto-resume music after 5s of silence

#### Volume Up/Down Buttons
- **Function**: Adjust MPD volume
- **Behavior**: Increase/decrease volume by 5%

#### Next/Previous Buttons
- **Function**: Skip tracks
- **Behavior**: Move to next/previous track in queue

## Configuration

### Required Settings

```bash
# Enable music system (required for all music features)
MUSIC_ENABLED=true
MPD_HOST=mpd  # or localhost
MPD_PORT=6600

# Enable media keys (required for button detection)
MEDIA_KEYS_ENABLED=true
# Optional: Filter by device name
MEDIA_KEYS_DEVICE_FILTER=Burr-Brown  # Your Anker speaker device name
```

### Music Auto-Sleep and Resume

```bash
# Put orchestrator to sleep while music is playing (default: true)
MUSIC_SLEEP_DURING_PLAYBACK=true

# Seconds of silence before auto-resuming music after wake (default: 5)
MUSIC_AUTO_RESUME_TIMEOUT_S=5

# Number of random tracks when play button pressed on empty queue (default: 50)
MUSIC_RANDOM_TRACK_COUNT=50
```

### Example Complete Configuration

```bash
# Music Control (MPD)
MUSIC_ENABLED=true
MPD_HOST=mpd
MPD_PORT=6600
MUSIC_SLEEP_DURING_PLAYBACK=true
MUSIC_AUTO_RESUME_TIMEOUT_S=5
MUSIC_RANDOM_TRACK_COUNT=50

# Media Keys (Hardware Button Detection)
MEDIA_KEYS_ENABLED=true
MEDIA_KEYS_DEVICE_FILTER=Burr-Brown
MEDIA_KEYS_CONTROL_MUSIC=true
```

## Workflows

### Workflow 1: Background Music

1. Say "play some rock music"
2. Music starts → Orchestrator goes to sleep
3. Music plays without generating transcripts
4. Say wake word ("hey mycroft")
5. Music stops → System wakes and listens
6. Say command or wait 5 seconds
7. After 5s silence → Music resumes automatically

### Workflow 2: Using Conference Speaker

1. Press **Play button** → 50 random tracks start playing
2. Music plays → Orchestrator sleeps
3. Press **Phone button** → Music stops, mic unmutes, system wakes
4. Speak your command
5. If no more input for 5s → Music resumes
6. Press **Mute button** → Microphone mutes (music continues)
7. Press **Mute button** again → Microphone unmutes
8. Press **Play button** → Music pauses
9. Press **Volume Up** → Volume increases

### Workflow 3: Voice Cut-In During TTS

1. Ask a question
2. TTS starts speaking answer
3. You start speaking (interrupt)
4. TTS stops immediately
5. Music stops immediately (if playing)
6. System captures your new input
7. After 5s of no input → Music resumes

## Button Summary

| Button | Action | Details |
|--------|--------|---------|
| **Mute** | Toggle microphone | Mutes/unmutes mic input only |
| **Play** | Smart play/stop | Adds 50 random tracks if queue empty |
| **Phone** | Trigger wake word | Stops audio, unmutes mic, wakes system |
| **Volume Up** | Increase volume | +5% MPD volume |
| **Volume Down** | Decrease volume | -5% MPD volume |
| **Next** | Skip forward | Next track in queue |
| **Previous** | Skip backward | Previous track |

## Technical Details

### Microphone Mute Implementation

- **AudioCapture**: Software mute (returns silence)
- **DuplexAudioIO**: Software mute (returns silence)
- Does NOT mute hardware - processes in software
- Zero latency toggle
- Persists until toggled

### Music State Tracking

- Polls MPD every 500ms for playback state
- Tracks: `is_playing`, `music_was_playing`, `music_paused_for_wake`
- Auto-resume timer starts when music stopped for wake
- Timer resets if voice activity detected
- Timer triggers after `MUSIC_AUTO_RESUME_TIMEOUT_S` seconds

### Wake Word and Cut-In Integration

**Wake Word Detection:**
```python
if wake_word_detected:
    1. Stop music (if playing)
    2. Set music_paused_for_wake = True
    3. Start auto-resume timer
    4. Wake system
```

**Voice Cut-In Detection:**
```python
if voice_detected_during_tts:
    1. Stop TTS playback
    2. Stop music (if playing)  
    3. Set music_paused_for_wake = True
    4. Start auto-resume timer
    5. Capture voice input
```

**Auto-Resume Logic:**
```python
if music_paused_for_wake and no_voice_activity:
    if timer >= MUSIC_AUTO_RESUME_TIMEOUT_S:
        1. Resume music playback
        2. Reset flags
        3. Return orchestrator to sleep (if configured)
```

### System States

- **ASLEEP + Music Playing**: Normal music playback (no transcription)
- **AWAKE + Music Stopped**: Listening for commands
- **AWAKE + No Input (5s)**: Auto-resume music → Return to ASLEEP

## Troubleshooting

### Music Doesn't Stop on Wake Word

**Check:**
1. `MUSIC_ENABLED=true` in `.env`
2. MPD is running: `docker ps | grep mpd`
3. Music manager initialized in logs
4. Wake word detector working properly

### Music Doesn't Auto-Resume

**Check:**
1. `MUSIC_AUTO_RESUME_TIMEOUT_S=5` is set
2. System is in IDLE or LISTENING state
3. No TTS playing or queued
4. Check logs for "Auto-resuming music after Xs"

### Mute Button Doesn't Work

**Check:**
1. `MEDIA_KEYS_ENABLED=true` in `.env`
2. Button event detected in logs
3. Capture device has `toggle_mute()` method
4. Check log: "Microphone muted/unmuted"

### Phone Button Doesn't Wake System

**Check:**
1. Phone button event detected: `"📞 Phone button pressed"`
2. `WAKE_WORD_ENABLED=true` (phone button requires wake system)
3. Wake click sound plays (if enabled)
4. System state changes to AWAKE in logs

### Play Button Doesn't Add Random Tracks

**Check:**
1. MPD library has music: `await music_manager.get_stats()`
2. `MUSIC_RANDOM_TRACK_COUNT=50` is set
3. Check logs: "Added X random tracks"
4. Library may need updating: Say "update library"

## Logs to Monitor

```bash
# Watch for music state changes
tail -f orchestrator_output.log | grep "🎵"

# Watch for button presses
tail -f orchestrator_output.log | grep "Media key pressed"

# Watch for microphone mute
tail -f orchestrator_output.log | grep "🎤"

# Watch for phone button wake
tail -f orchestrator_output.log | grep "📞"

# Watch for wake word
tail -f orchestrator_output.log | grep "Wake word detected"

# Watch for auto-resume
tail -f orchestrator_output.log | grep "Auto-resuming"
```

## Performance Impact

- **Music state polling**: 500ms interval (minimal CPU)
- **Microphone mute**: Zero latency (software toggle)
- **Button detection**: Event-driven (no polling)
- **Auto-resume timer**: Piggybacks on main loop (no threads)

## Customization

### Change Auto-Resume Timeout

Edit `.env`:
```bash
MUSIC_AUTO_RESUME_TIMEOUT_S=10  # Wait 10 seconds instead of 5
```

### Change Random Track Count

Edit `.env`:
```bash
MUSIC_RANDOM_TRACK_COUNT=100  # Add 100 tracks instead of 50
```

### Disable Music Sleep

Edit `.env`:
```bash
MUSIC_SLEEP_DURING_PLAYBACK=false  # Keep transcription active during music
```

### Change Volume Step

Edit `orchestrator/main.py` in media key callback:
```python
elif event.key == "volume_up":
    asyncio.create_task(music_manager.increase_volume(10))  # Change 5 to 10
```

## Advanced: Custom Button Actions

Edit the `on_media_key_press()` callback in `orchestrator/main.py`:

```python
def on_media_key_press(event: MediaKeyEvent):
    # Add custom logic here
    if event.key == "mute" and some_condition:
        # Do something custom
        pass
    
    # Original logic follows...
```

## FAQ

**Q: Can I use this with Bluetooth speakers?**  
A: Yes, if the speaker buttons send HID events via Bluetooth.

**Q: Will music playback interfere with voice detection?**  
A: No, when `MUSIC_SLEEP_DURING_PLAYBACK=true`, the system stops transcribing during music.

**Q: Can I use voice commands while music is playing?**  
A: Yes, say the wake word and music will stop immediately to listen.

**Q: What if I want to keep music playing during voice commands?**  
A: Set `MUSIC_SLEEP_DURING_PLAYBACK=false`, but music may interfere with STT.

**Q: Can I use these features without a conference speaker?**  
A: Yes, voice wake word and auto-resume work without hardware buttons.

**Q: Does muting the microphone affect music volume?**  
A: No, microphone mute only affects voice input, not audio output.

**Q: Can I disable auto-resume?**  
A: Set `MUSIC_AUTO_RESUME_TIMEOUT_S=0` (not tested, may need code change).

## See Also

- [MEDIA_KEYS_GUIDE.md](MEDIA_KEYS_GUIDE.md) - Hardware button detection setup
- [MUSIC_CONTROL_PLAN.md](MUSIC_CONTROL_PLAN.md) - MPD integration details
- [WAKE_WORD_CONFIG.md](WAKE_WORD_CONFIG.md) - Wake word configuration
