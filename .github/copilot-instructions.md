# Copilot Instructions for OpenClaw Voice Python

## Development Workflow

### After Making Code Changes

**DO NOT automatically restart the orchestrator after making code changes.**
The user will manually restart when ready.

To restart and view logs when requested:
```bash
./run_voice_demo.sh
tail -f orchestrator_output.log
```

### After Modifying Docker Containers

**If changes are made to the Piper or Whisper containers, they must be rebuilt:**
```bash
# Rebuild specific container
docker-compose build piper
docker-compose build whisper

# Or rebuild all containers
docker-compose build

# Then restart the services
docker-compose up -d
```

## Project Context

This is a Python-based voice orchestrator that integrates:
- STT (Speech-to-Text) via Whisper
- TTS (Text-to-Speech) via Piper
- VAD (Voice Activity Detection) via WebRTC/Silero
- Optional wakeword detection via OpenWakeWord
- Optional emotion detection via SenseVoice

The main orchestrator runs locally, while Whisper and Piper run in Docker containers.

## Known Issues

### Wake Word Sensitivity and Audio Feedback
The wake word detector (OpenWakeWord with "hey_mycroft" model) has had sensitivity issues:
- **Audio feedback loop**: Wake click sound was being picked up by microphone, triggering more detections
  - Wake click sound playback is now DISABLED to prevent this
  - No false positives on first wake (before any audio feedback played)
  - Constant false positives after sleep/wake cycle (when feedback sounds had played)
- **Immediate re-detection after timeout**: Model internal state was carrying over
  - Now calls `reset_state()` on wake detector when going to sleep to clear prediction buffer
  - 1000ms cooldown after sleep before wake detection re-enables
- Current configuration: `WAKE_WORD_CONFIDENCE=0.95`, `OPENWAKEWORD_MODEL_PATH=hey_mycroft`
- Model switched from "alexa" to "hey_mycroft" to reduce false positives
- Alternative models available: `hey_jarvis`, `timer`, `weather`
