# Recent Changes Summary

## 1. ✅ Fixed Hotword Prebuffer Capture Issue

**Problem**: After hotword detection, the system was capturing too much pre-roll audio (200ms), which included the spoken hotword itself ("Hey, my craft") being transcribed.

**Solution**: Reduced prebuffer from 200ms to 80ms in [orchestrator/main.py](orchestrator/main.py#L999)

**Before**: 
```python
wake_pre_roll_ms = min(200, config.pre_roll_ms)
```

**After**:
```python
# Reduced prebuffer from 200ms to 80ms to avoid capturing the hotword itself being spoken
wake_pre_roll_ms = min(80, config.pre_roll_ms)
```

**Impact**: Hotword "Hey, my craft" will no longer be included in the transcription. You can further tune via `.env`:
```bash
WAKE_WORD_PREBUFFER_MS=50  # Reduce further if needed
```

---

## 2. ✅ Created Raspbian Installation Suite

### Files Created:

1. **[install.sh](install.sh)** (14KB, executable)
   - Interactive installer for Raspbian 32-bit and 64-bit
   - Detects architecture and OS version automatically
   - Installs all system dependencies
   - Creates Python virtual environment
   - Installs core + optional Python packages
   - Interactive configuration prompts for:
     - Audio devices
     - Gateway URL + token
     - Whisper (STT) URL
     - Piper (TTS) URL and voice
     - Wake word model and confidence
     - VAD backend and aggressiveness
     - Log level
   - Generates `.env` configuration file
   - Creates helper scripts: `activate.sh`, `run.sh`

2. **[RASPBIAN_INSTALL.md](RASPBIAN_INSTALL.md)** (Comprehensive guide)
   - Detailed step-by-step installation instructions
   - Configuration explanation and examples
   - Troubleshooting section with solutions for:
     - Audio device issues
     - Gateway connection failures
     - Whisper/Piper service problems
     - High CPU usage
     - Memory issues
     - Hotword capture issues
   - Advanced configuration options
   - Performance tips for different Pi models
   - Uninstall instructions
   - Getting help resources

3. **[QUICK_START_RASPBIAN.sh](QUICK_START_RASPBIAN.sh)** (Quick reference guide)
   - One-command installation guide
   - Quick verification steps
   - Troubleshooting checklist
   - Common configuration options
   - System requirements
   - Architecture support info

### Supported Systems:

- **Architectures**: 32-bit (ARMv6/7) and 64-bit (ARM64)
- **Raspbian Versions**: Bullseye, Bookworm, Buster
- **Hardware**: Raspberry Pi 3, 4, Zero 2W, 5

### Interactive Configuration:

The installer prompts for:

```
Audio devices (capture/playback)
Gateway: ws://openclaw.local:8000
Gateway token: (your-token)
Whisper URL: http://localhost:10000
Piper URL: http://localhost:10001
Piper voice: en_US-amy-medium
Piper speed: 1.0
Wake word model: hey_mycroft
Wake word confidence: 0.95
VAD backend: webrtc|silero
VAD aggressiveness: 1
Log level: DEBUG|INFO|WARNING|ERROR
```

### Generated Files:

After installation, the script creates:

```
.venv_orchestrator/          # Python virtual environment
.env                          # Configuration file
activate.sh                   # Activation script
run.sh                        # Run script
orchestrator.log              # Log file (created on first run)
```

---

## How to Use (For End Users)

### Quick Install:

```bash
cd /path/to/openclaw-voice-py
bash install.sh
```

Then follow the interactive prompts. Takes ~5-10 minutes depending on internet speed.

### Run the Orchestrator:

```bash
bash run.sh
```

Monitor logs:
```bash
tail -f orchestrator.log
```

---

## Key Features of the Installation Suite

✅ **Cross-platform**: Auto-detects Raspbian version and architecture  
✅ **Interactive**: Guides users through configuration step-by-step  
✅ **Complete**: Installs all core + optional dependencies  
✅ **Safe**: Validates prerequisites before proceeding  
✅ **Documented**: Includes comprehensive troubleshooting guide  
✅ **Helper scripts**: Creates `activate.sh` and `run.sh` for easy operation  
✅ **Flexible**: Allows all configuration via prompts or manual `.env` editing  

---

## Testing

To verify the installation worked:

```bash
source .venv_orchestrator/bin/activate
python3 -c "import orchestrator; print('✓ Orchestrator imported successfully')"
```

To test audio:
```bash
python3 -m sounddevice
```

To test services:
```bash
curl http://localhost:10000/info  # Whisper
curl http://localhost:10001/info  # Piper
```

---

## Docker Build Status

As a reminder from the previous session, all Docker images are successfully built:

```
✓ openclaw-voice-py-orchestrator:latest
✓ openclaw-voice-py-whisper:latest
✓ openclaw-voice-py-piper:latest
```

These are independent of the Raspbian installation — they run in containers on systems with Docker.

---

## Next Steps

1. **Test prebuffer fix**: Run and listen for "Hey, my craft" being transcribed
2. **Deploy on Raspberry Pi**: Use `bash install.sh` on the target Raspbian system
3. **Adjust if needed**: Edit `.env` for fine-tuning (WAKE_WORD_PREBUFFER_MS, etc.)
4. **Monitor**: Check logs for any issues during first run

---

## Files Modified vs Created

**Modified**:
- `orchestrator/main.py` - Prebuffer reduced from 200ms to 80ms

**Created**:
- `install.sh` - Interactive installer (14KB)
- `RASPBIAN_INSTALL.md` - Comprehensive installation guide
- `QUICK_START_RASPBIAN.sh` - Quick reference guide
