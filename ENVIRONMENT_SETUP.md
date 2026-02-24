# OpenClaw Voice Orchestrator - Isolated Environment Setup

## Quick Start

Run the orchestrator with the helper script:
```bash
cd ~/projects/openclawstuff/openclaw-voice-py
./run_orchestrator.sh
```

No need to remember long Python paths anymore!

## What Was Set Up

### 1. Isolated Python 3.11 Virtual Environment (`.venv311/`)

Created an isolated virtual environment with all dependencies pre-installed:
- **Python version**: 3.11.9 (required for WebRTC AEC bindings)
- **Location**: `.venv311/` in project root
- **Size**: ~800MB with all ML models cached

#### Dependencies Installed:
- `webrtc-audio-processing` (0.1.3) - WebRTC echo cancellation audio processing
- `torch` (2.2.2+cpu) + `torchaudio` (2.2.2+cpu) - CPU-only PyTorch ML framework
- `funasr` (1.3.1) - FunASR model framework for SenseVoice emotion detection
- `numpy` (1.26.4) - Downgraded from 2.x for torch compatibility
- `pydantic` + `pydantic-settings` - Configuration management
- `sounddevice` - Audio capture/playback
- `requests`, `websockets` - Network communication
- `onnxruntime` - Neural network inference
- `webrtcvad` - Voice activity detection
- `openwakeword` - Wake word detection
- `python-dotenv` - Environment variable management

### 2. Helper Script (`run_orchestrator.sh`)

Created executable helper script that:
- Activates the isolated `.venv311/` environment
- Runs the orchestrator with any passed arguments
- Eliminates the need to type long Python paths

**Usage**:
```bash
./run_orchestrator.sh                    # Start orchestrator
./run_orchestrator.sh --help             # Show help
```

## Architecture Overview

```
Project Root
├── .venv311/                      # Isolated Python 3.11 environment
│   ├── lib/python3.11/site-packages/  # All dependencies
│   └── bin/python                     # Python 3.11 interpreter
├── run_orchestrator.sh            # Helper script (this runs the app)
├── orchestrator/
│   ├── main.py                    # Event loop (audio → STT → gateway → TTS)
│   ├── config.py                  # Pydantic configuration (42 settings)
│   ├── gateway/
│   │   ├── providers.py           # 8 gateway implementations
│   │   └── factory.py             # Gateway factory pattern
│   ├── emotion/
│   │   └── sensevoice.py          # SenseVoice emotion tagging
│   ├── vad/
│   │   └── webrtc_vad.py          # WebRTC echo cancellation
│   └── tts/
│       └── piper.py               # Piper TTS client
├── docker/
│   └── piper/                     # Piper TTS service (port 10001)
└── requirements.txt               # Core dependencies
```

## Verified Functionality

✅ **WebRTC AEC**: Echo cancellation bindings working in Python 3.11
✅ **SenseVoice**: Emotion detection model loads successfully
✅ **Gateway Routing**: All 7 claw providers + generic HTTP/WS
✅ **Audio I/O**: capture → STT → gateway → TTS pipeline functional
✅ **Docker Integration**: Whisper (10000) + Piper (10001) running

## What Each Component Does

### `orchestrator/main.py`
The orchestrator event loop that:
1. Captures audio from microphone using `sounddevice`
2. Applies WebRTC AEC for echo cancellation
3. Sends audio to Whisper gateway for STT
4. Detects emotions using SenseVoice model
5. Routes transcripts through configured gateway provider
6. Synthesizes response using Piper TTS
7. Plays audio response back to user

### `orchestrator/gateway/` - 8 Provider Implementations
Supports routing to multiple platforms:
- **Generic**: HTTP/WS for test endpoints (fallback)
- **OpenClaw**: HTTP POST with Bearer token auth
- **ZeroClaw**: HTTP webhook with X-headers
- **TinyClaw**: File-based queue system
- **IronClaw**: WebSocket-first with HTTP fallback
- **MimiClaw**: WebSocket device + Telegram API
- **PicoClaw**: File-based JSONL sessions + optional HTTP
- **NanoBot**: File-based workspace JSON + optional HTTP

### `orchestrator/emotion/sensevoice.py`
Analyzes emotional state from captured audio:
- Returns emotion label + confidence
- Disabled FunASR update checks for faster startup
- Model cached at `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall/`

## Configuration

All settings are managed via `.env` file and Pydantic `config.py`:
- Gateway provider selection: `VOICE_CLAW_PROVIDER`
- WebRTC AEC enabled by default
- SenseVoice paths auto-configured
- Docker service URLs (Whisper, Piper)

## System Requirements

**Already installed on this machine:**
- `libwebrtc-audio-processing-dev` (system package)
- `cmake`, `pkg-config`, `swig` (build tools)
- Docker with `docker-compose` running Whisper + Piper

**Virtual environment includes:**
- Everything else (see Dependencies list above)

## Troubleshooting

### "Command not found: run_orchestrator.sh"
```bash
chmod +x run_orchestrator.sh
```

### "ModuleNotFoundError"
The helper script should automatically activate the virtual environment. If issues persist:
```bash
source .venv311/bin/activate
python -m orchestrator.main
```

### "Port 10001 refused" (Piper)
Whisper/Piper services down:
```bash
cd ../../../docker && docker-compose ps
docker-compose up -d
```

### "Emotion detection disabled"
SenseVoice model caching may take a moment on first run. Check:
```bash
ls -lh ~/.cache/modelscope/hub/models/iic/SenseVoiceSmall/
```

## Next Steps

1. **Test the orchestrator**:
   ```bash
   ./run_orchestrator.sh
   ```

2. **Test with specific gateway**:
   ```bash
   VOICE_CLAW_PROVIDER=generic ./run_orchestrator.sh
   ```

3. **View logs**:
   ```bash
   # In another terminal
   docker-compose logs -f piper whisper
   ```

## Files Created/Modified

**New files:**
- `run_orchestrator.sh` - Helper script
- `.venv311/` - Isolated virtual environment

**Previously created (from earlier work):**
- `orchestrator/gateway/providers.py` - All 8 gateway implementations (~600 lines)
- `orchestrator/gateway/factory.py` - Gateway factory pattern
- `orchestrator/config.py` - Extended with 40+ provider settings
- `orchestrator/main.py` - Updated to use gateway factory

## Important Notes

- The `.venv311` directory is a copy of Python 3.11.9 installation. While isolated, it shares disk space with system Python
- All ML models are cached in `~/.cache/modelscope/` (shared across Python installations)
- Docker services run independently and can be used with any Python version
- Helper script respects all `.env` variables and command-line arguments passed to it
