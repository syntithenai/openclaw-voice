# Music Integration and Media Keys - Implementation Summary

## What Was Implemented

### 1. ✅ Music Playback State Management

**Auto-Sleep During Music:**
- Orchestrator monitors MPD playback state every 500ms
- When music starts → System goes to ASLEEP (stops transcription)
- Prevents false transcriptions from music

**Wake on Command:**
- Wake word detected → Stop music immediately
- Voice cut-in detected → Stop music and TTS immediately
- System captures voice input

**Auto-Resume:**
- After wake/cut-in, if no voice activity for 5 seconds (configurable)
- Music automatically resumes playing
- System returns to sleep mode

### 2. ✅ Conference Speaker Button Integration

**Mute Button:**
- Toggle microphone mute/unmute
- Software-level mute (returns silence to pipeline)
- Works with both AudioCapture and DuplexAudioIO backends

**Play Button:**
- Smart play: If queue empty → Add 50 random tracks and play
- If playing → Pause
- If paused → Resume

**Phone Button:**
- Triggers wake word sequence:
  1. Stop music/TTS playback
  2. Play wake notification sound
  3. Unmute microphone (if muted)
  4. Wake system and start listening
  5. Auto-resume after 5s of silence

**Volume/Navigation Buttons:**
- Volume Up/Down: Adjust MPD volume by 5%
- Next/Previous: Skip tracks

## Files Modified

### Core Modules

1. **orchestrator/audio/capture.py**
   - Added `_muted` flag
   - Added `set_muted()`, `is_muted()`, `toggle_mute()` methods
   - Modified callback to return silence when muted

2. **orchestrator/audio/duplex.py**
   - Added `_muted` flag
   - Added mute methods (same as capture.py)
   - Modified callback to return silence when muted

3. **orchestrator/music/manager.py**
   - Added `is_playing()` - Check if music is currently playing
   - Added `is_paused()` - Check if music is paused
   - Added `get_playback_state()` - Get 'play', 'pause', or 'stop'
   - Added `toggle_playback()` - Toggle play/pause
   - Added `get_queue_length()` - Get queue size
   - Added `add_random_tracks(count)` - Add N random tracks
   - Added `smart_play(count)` - Smart play/pause with auto-random tracks
   - Added `increase_volume()` / `decrease_volume()` - Wrapper methods

4. **orchestrator/config.py**
   - Added `music_sleep_during_playback` (default: True)
   - Added `music_auto_resume_timeout_s` (default: 5)
   - Added `music_random_track_count` (default: 50)

5. **orchestrator/main.py**
   - Added music state tracking variables:
     - `music_was_playing` - Track if music is currently playing
     - `music_paused_for_wake` - Track if music stopped for wake/cut-in
     - `music_auto_resume_timer` - Timer for auto-resume
   - Added music state monitoring loop (every 500ms)
   - Updated wake word detection to stop music
   - Updated voice cut-in detection to stop music and TTS
   - Updated media key callback with new button functions
   - Added auto-resume logic

### Configuration

6. **.env**
   - Added `MUSIC_SLEEP_DURING_PLAYBACK=true`
   - Added `MUSIC_AUTO_RESUME_TIMEOUT_S=5`
   - Added `MUSIC_RANDOM_TRACK_COUNT=50`

### Documentation

7. **MUSIC_INTEGRATION_GUIDE.md** (new)
   - Complete guide to all music integration features
   - Button functions, workflows, troubleshooting
   - Configuration examples

8. **MUSIC_INTEGRATION_SUMMARY.md** (new)
   - This file - implementation summary

## Key Behaviors

### Scenario 1: Background Music
```
1. Music playing → Orchestrator ASLEEP
2. Say wake word → Music stops, system AWAKE
3. Say command OR wait 5s
4. After 5s → Music resumes, system ASLEEP
```

### Scenario 2: Conference Speaker Control
```
1. Press Play → 50 random tracks play, system ASLEEP
2. Press Phone → Music stops, system AWAKE
3. Speak command OR wait 5s
4. After 5s → Music resumes
5. Press Mute → Mic mutes (music continues)
6. Press Mute → Mic unmutes
```

### Scenario 3: Voice Cut-In
```
1. TTS speaking answer
2. User starts speaking → TTS stops, music stops
3. System captures new input
4. After processing OR 5s silence → Music resumes
```

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `MUSIC_ENABLED` | false | Enable MPD music control |
| `MUSIC_SLEEP_DURING_PLAYBACK` | true | Sleep orchestrator during music |
| `MUSIC_AUTO_RESUME_TIMEOUT_S` | 5 | Seconds before auto-resume |
| `MUSIC_RANDOM_TRACK_COUNT` | 50 | Random tracks for empty queue |
| `MEDIA_KEYS_ENABLED` | false | Enable button detection |
| `MEDIA_KEYS_DEVICE_FILTER` | "" | Filter by device name |
| `MEDIA_KEYS_CONTROL_MUSIC` | true | Allow buttons to control music |

## Button Mapping

| Button | Function |
|--------|----------|
| Mute | Toggle microphone mute/unmute |
| Play | Smart play/stop (adds random tracks if queue empty) |
| Phone | Trigger wake word (stop audio, unmute, wake, listen) |
| Volume Up | Increase MPD volume +5% |
| Volume Down | Decrease MPD volume -5% |
| Next | Skip to next track |
| Previous | Skip to previous track |

## Testing Checklist

- [ ] Music starts → Orchestrator goes to sleep
- [ ] Wake word → Music stops immediately
- [ ] No command for 5s → Music resumes automatically
- [ ] Mute button → Microphone mutes/unmutes
- [ ] Play button (empty queue) → Adds random tracks and plays
- [ ] Play button (playing) → Pauses
- [ ] Phone button → Stops audio, wakes system, listens
- [ ] Voice cut-in → Stops TTS and music
- [ ] Volume buttons → Adjust MPD volume
- [ ] Next/Previous buttons → Skip tracks

## Known Limitations

1. **Music state polling**: 500ms interval (acceptable for this use case)
2. **Microphone mute**: Software-only (doesn't disable hardware)
3. **Auto-resume**: Only works in IDLE or LISTENING states
4. **Phone button**: Requires `WAKE_WORD_ENABLED=true` to function
5. **Random tracks**: Requires music in MPD library

## Future Enhancements

Potential improvements not yet implemented:

- Configurable button mappings
- Remember previous volume before mute
- Smart volume ducking (lower music during voice, not stop)
- Playlist-aware random selection
- Hardware LED indicators for mute state
- Multiple auto-resume timeout scenarios

## Logs to Watch

```bash
# Music state changes
tail -f orchestrator_output.log | grep "🎵"

# Button presses
tail -f orchestrator_output.log | grep "Media key pressed"

# Microphone state
tail -f orchestrator_output.log | grep "🎤"

# Phone button
tail -f orchestrator_output.log | grep "📞"

# Complete wake sequence
tail -f orchestrator_output.log | grep -E "🎵|📞|🎤|Wake word"
```

## Quick Start

1. **Enable features:**
   ```bash
   # Edit .env
   MUSIC_ENABLED=true
   MEDIA_KEYS_ENABLED=true
   MEDIA_KEYS_DEVICE_FILTER=Burr-Brown  # Your device name
   MUSIC_SLEEP_DURING_PLAYBACK=true
   MUSIC_AUTO_RESUME_TIMEOUT_S=5
   ```

2. **Add yourself to input group** (for button access):
   ```bash
   sudo usermod -a -G input $USER
   # Log out and back in
   ```

3. **Test button detection:**
   ```bash
   sudo python3 find_anker_device.py
   # Press buttons to verify
   ```

4. **Start orchestrator:**
   ```bash
   ./run_voice_demo.sh
   ```

5. **Test workflow:**
   - Press **Play** button → Music should start
   - Say **wake word** → Music should stop
   - Wait 5 seconds → Music should resume
   - Press **Mute** → Mic should mute
   - Press **Phone** → Should trigger wake sequence

## Support

See these files for more details:
- **MUSIC_INTEGRATION_GUIDE.md** - Complete feature guide
- **MEDIA_KEYS_GUIDE.md** - Button detection setup
- **MEDIA_KEYS_QUICKSTART.md** - Quick setup guide

For issues, check logs and verify configuration matches examples above.
