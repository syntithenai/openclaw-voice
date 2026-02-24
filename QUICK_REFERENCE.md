# Quick Reference Guide

## 🚀 Common Commands

### Start Services

```bash
# Start fake gateway (for testing)
python -m orchestrator.tools.fake_gateway

# Start orchestrator with audio capture
python -m orchestrator.main

# Start all services with Docker
docker-compose up -d
docker-compose logs -f

# Stop Docker services
docker-compose down
```

### Testing

```bash
# Run complete end-to-end tests
python e2e_test.py

# Verify setup
python verify_setup.py

# Test fake gateway manually
curl -X POST http://localhost:18901/api/short
curl http://localhost:18901/health
```

### Configuration

```bash
# View current configuration
grep -E "VAD_TYPE|AUTO_DOWNLOAD|GATEWAY|MODELS_DIR" .env

# Edit configuration
nano .env
# or
vim .env

# Reload configuration (restart orchestrator)
# Changes to .env are picked up on next run
```

### Model Management

```bash
# List model directories
ls -la docker/{silero,wakeword,emotion}-models/

# Check disk usage
du -sh docker/*/

# Clear model cache (will re-download on next run)
rm -rf docker/{silero,wakeword,emotion}-models/*
```

### Debugging

```bash
# Check Python environment
python --version
pip list | grep -E "pydantic|sounddevice"

# Enable verbose logging
# Edit orchestrator/main.py or
export LOG_LEVEL=DEBUG

# View logs
tail -f orchestrator_output.log
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `.env` | Configuration (auto-loaded by BaseSettings) |
| `orchestrator/config.py` | Pydantic BaseSettings config schema |
| `orchestrator/main.py` | Main event loop (audio → VAD → STT → TTS) |
| `orchestrator/tools/fake_gateway.py` | Test endpoint server (:18901) |
| `orchestrator/audio/capture.py` | Audio input via sounddevice |
| `orchestrator/vad/webrtc_vad.py` | Speech detection (VAD) |
| `orchestrator/audio/buffer.py` | Ring buffer for pre-roll |
| `docker-compose.yml` | Docker service orchestration |
| `e2e_test.py` | End-to-end test suite |
| `verify_setup.py` | Setup verification script |

## 🔌 Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Whisper | :10000 | Speech-to-Text HTTP API |
| Piper | :10001 | Text-to-Speech HTTP API |
| Fake Gateway | :18901 | Test endpoints |
| Orchestrator | :18901 | Health check |

## 📋 Environment Variables

### Core Audio
```
AUDIO_SAMPLE_RATE=16000
AUDIO_FRAME_MS=20
AUDIO_CAPTURE_DEVICE=default
AUDIO_PLAYBACK_DEVICE=default
```

### VAD (Voice Activity Detection)
```
VAD_TYPE=webrtc                    # or "silero"
SILERO_AUTO_DOWNLOAD=true
SILERO_MODEL_CACHE_DIR=docker/silero-models
```

### Wakeword Detection
```
WAKE_WORD_ENABLED=false            # Set to true to enable
OPENWAKEWORD_AUTO_DOWNLOAD=true
OPENWAKEWORD_MODELS_DIR=docker/wakeword-models
```

### Emotion Detection
```
EMOTION_ENABLED=true
EMOTION_AUTO_DOWNLOAD=true
EMOTION_MODELS_DIR=docker/emotion-models
```

### Services
```
WHISPER_URL=http://localhost:10000
PIPER_URL=http://localhost:10001
GATEWAY_WS_URL=ws://localhost:18900
GATEWAY_AGENT_ID=test-agent
GATEWAY_AUTH_TOKEN=test-token
```

## 🧪 Test Endpoints

### Fake Gateway (for testing)

**Short Response:**
```bash
curl -X POST http://localhost:18901/api/short
```
Response:
```json
{
  "ok": true,
  "text": "That's correct.",
  "emotion": "neutral",
  "duration_ms": 500
}
```

**Long Response:**
```bash
curl -X POST http://localhost:18901/api/long
```
Response:
```json
{
  "ok": true,
  "text": "I understand you're asking about that. Let me explain in detail...",
  "emotion": "helpful",
  "duration_ms": 4500
}
```

**Health Check:**
```bash
curl http://localhost:18901/health
```
Response:
```json
{
  "status": "ok"
}
```

## 🏗️ Architecture

```
Input Audio Stream
    ↓
[Audio Capture] → sounddevice (PortAudio)
    ↓
[WebRTC VAD] → Speech detection (20ms frames)
    ↓
[Ring Buffer] → Pre-roll storage (2000ms)
    ↓
[Optional: Wakeword] → OpenWakeWord detection
    ↓
[Whisper STT] → Speech-to-text (HTTP :10000)
    ↓
[Optional: Emotion] → SenseVoice emotion detection
    ↓
[Fake/Real Gateway] → Process command
    ↓
[Piper TTS] → Text-to-speech (HTTP :10001)
    ↓
[Audio Playback] → Speaker output
```

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Install requirements
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

### Issue: Audio device not found
```bash
# Check available devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Set in .env
AUDIO_CAPTURE_DEVICE=2  # Use device index
```

### Issue: Fake gateway not responding
```bash
# Start fake gateway if not running
python -m orchestrator.tools.fake_gateway &

# Verify it's running
curl http://localhost:18901/health
```

### Issue: Models not auto-downloading
```bash
# Check config
grep AUTO_DOWNLOAD .env

# Manually download (example: Silero)
python -c "from orchestrator.vad.silero import SileroVAD; SileroVAD()"
```

### Issue: Permission denied on audio device
```bash
# On Linux, add user to audio group
sudo usermod -a -G audio $USER

# Then logout and log back in
```

## 📊 Performance Tips

1. **Use WebRTC VAD** (default) instead of Silero for faster performance
2. **Disable unused features**:
   - Wakeword: `WAKE_WORD_ENABLED=false`
   - Emotion: `EMOTION_ENABLED=false`
3. **Pre-warm services** with a dummy request before recording
4. **Cache models** to avoid download delays on first run
5. **Use local gateway** (fake or real) instead of remote when testing

## 📚 Documentation

- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Complete setup guide
- **[SESSION_SUMMARY.md](SESSION_SUMMARY.md)** - This session's work
- **[README.md](README.md)** - Project overview
- **[orchestrator/config.py](orchestrator/config.py)** - Config schema

## 🔗 Related Projects

- **OpenClaw**: Main Node.js gateway - `/home/stever/projects/openclawstuff/openclaw/`
- **Docker Services**: Whisper, Piper in `./docker/`
- **Tests**: E2E tests in `e2e_test.py`

## ✅ Status Checklist

- [x] Models organized in docker/
- [x] Auto-download enabled for all models
- [x] Gateway authentication configured
- [x] Fake test endpoints available
- [x] Error handling implemented
- [x] Docker compose configured
- [x] End-to-end tests passing
- [x] Configuration validated
- [x] Ready for deployment

---

**Last Updated**: 2024-02-24
**Status**: ✅ Production Ready
