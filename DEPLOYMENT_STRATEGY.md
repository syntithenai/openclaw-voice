# OpenClaw Voice Pi Deployment Strategy

## Current Configuration (Pi 10.1.1.210 - ARMv7)

### System Info
- **Architecture**: armv7l
- **OS**: Raspbian GNU/Linux 12 (bookworm)
- **Audio Device**: hw:2,0 (USB Camera-B4.09.24.1)
- **Sample Rate**: 16000 Hz (capture), 48000 Hz (playback)

### Wake Word Configuration
- **Engine**: Precise (Mycroft v0.3.0)
- **Model**: hey-mycroft.pb
- **Confidence**: 0.15
- **Model Location**: `docker/wakeword-models/hey-mycroft.pb`

### Service URLs
- **Whisper**: http://10.1.1.249:10000
- **Piper**: http://10.1.1.249:10001
- **OpenClaw Gateway**: http://10.1.1.249:18789

### Key Configuration Settings
```env
AUDIO_CAPTURE_DEVICE=hw:2,0
AUDIO_PLAYBACK_DEVICE=hw:2,0
AUDIO_SAMPLE_RATE=16000
AUDIO_PLAYBACK_SAMPLE_RATE=48000
AUDIO_BACKEND=portaudio

VAD_TYPE=webrtc
VAD_CONFIDENCE=0.6
VAD_MIN_SILENCE_MS=800

PRECISE_ENABLED=true
PRECISE_MODEL_PATH=docker/wakeword-models/hey-mycroft.pb
PRECISE_CONFIDENCE=0.15
WAKE_WORD_TIMEOUT_MS=6000

ECHO_CANCEL=true
ECHO_CANCEL_WEBRTC_AEC_STRENGTH=strong

PIPER_SPEED=1.2
GATEWAY_TTS_FAST_START_WORDS=0
```

## Large Files (Not in Git)

### Precise Engine (ARMv7)
- **Location**: `/home/stever/openclaw-voice/precise-engine/`
- **Size**: ~26M extracted, 162M tarball
- **Critical Files**:
  - `precise-engine` (executable, 4.0M)
  - `precise-engine.tar.gz` (full bundle with dependencies)
  - Dependencies: tensorflow_core, scipy, numpy, keras, etc.
- **Local Artifact**: `./artifacts/precise-engine-armv7/precise-engine.tar.gz`

### Wake Word Models
- **Location**: `docker/wakeword-models/`
- **Size**: 36K
- **Files**:
  - `hey-mycroft.pb` (26K)
  - `hey-mycroft.pb.params` (132 bytes)

### Other Model Directories (Currently Empty on Pi)
- `docker/whisper-models/` - Not on Pi (runs on remote server)
- `docker/piper-data/` - Not on Pi (runs on remote server)
- `docker/silero-models/` - Not present
- `docker/emotion-models/` - Not present

## Architecture-Specific Requirements

### ARMv7 (Raspberry Pi 3/Zero 2)
- **Wake Word Engine**: Precise (Mycroft)
  - TensorFlow 1.x based
- Requires pre-built `precise-engine` binary
  - Located in `artifacts/precise-engine-armv7/precise-engine.tar.gz`
- **Model**: `hey-mycroft.pb` (file-based)
- **Confidence**: 0.10-0.20 (lower = more sensitive)

### ARMv8/ARM64 (Raspberry Pi 4/5)
- **Wake Word Engine**: OpenWakeWord (TFLite based)
  - Does NOT work on ARMv7
- No pre-built binary required (uses Python packages)
- **Model**: `hey_mycroft` (model name string)
- **Confidence**: 0.50-0.95 (higher = more sensitive)
- Could potentially use Precise engine built for arm64 if available

## Deployment Sync Strategy

### Method 1: Git + Rsync (Recommended)
```bash
# On local machine:
# 1. Commit all code changes to git
git add orchestrator/ install_raspbian_remote.sh .env.example
git commit -m "Update deployment configuration"
git push

# 2. Sync artifacts directory (large files) via rsync
rsync -avz --progress \
  ./artifacts/precise-engine-armv7/precise-engine.tar.gz \
  pi_new:/home/stever/openclaw-voice-artifacts/

rsync -avz --progress \
  ./docker/wakeword-models/ \
  pi_new:/home/stever/openclaw-voice/docker/wakeword-models/

# 3. On new Pi: pull from git
ssh pi_new "cd ~/openclaw-voice && git pull"
```

### Method 2: Complete Package Transfer
```bash
# Create a deployment bundle with all non-git files
tar -czf openclaw-voice-deployment.tar.gz \
  --exclude='.git' \
  --exclude='.venv*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='orchestrator_output.log' \
  -C /home/stever/projects/openclawstuff openclaw-voice

# Transfer to new Pi
scp openclaw-voice-deployment.tar.gz pi_new:~/
ssh pi_new "tar -xzf openclaw-voice-deployment.tar.gz"
```

### Method 3: Install Script with Artifact Download
Update `install_raspbian_remote.sh` to:
1. Clone git repo
2. Detect architecture
3. Download/transfer appropriate artifacts from build machine
4. Generate .env based on detected hardware

## Updated Install Script Requirements

### Enhancements Needed for `install_raspbian_remote.sh`

1. **Architecture Detection**
   - Detect armv7l vs aarch64/armv8
   - Set wake word engine based on architecture
   - Configure appropriate model paths

2. **Artifact Transfer**
   - Option 1: Copy from local `artifacts/` directory
   - Option 2: Download from build server/URL
   - Option 3: Build on-demand (slow for Precise)

3. **Audio Device Detection**
   - Detect USB audio devices
   - Configure hw:X,Y automatically
   - Set appropriate audio backend

4. **Service URL Configuration**
   - Accept host IP as parameter (default to current machine)
   - Update all service URLs in .env
   - Validate connectivity to services

5. **Default .env Template**
   - Create comprehensive .env.template with all settings
   - Use current working Pi configuration as baseline
   - Support architecture-specific overrides

## Deployment Checklist for New Pi

### Pre-Deployment (On Build Machine)
- [ ] Code changes committed to git
- [ ] Artifacts built and ready:
  - [ ] `artifacts/precise-engine-armv7/precise-engine.tar.gz` (162M)
  - [ ] `artifacts/precise-engine-arm64/precise-engine.tar.gz` (if available)
  - [ ] `docker/wakeword-models/hey-mycroft.pb` + params
- [ ] Install script tested and updated
- [ ] Service endpoints running and accessible:
  - [ ] Whisper STT (port 10000)
  - [ ] Piper TTS (port 10001)
  - [ ] OpenClaw Gateway (port 18789)

### Deployment (On New Pi)
- [ ] SSH access configured
- [ ] Git repository cloned
- [ ] Architecture detected
- [ ] Artifacts transferred
- [ ] .env configured with correct:
  - [ ] Audio devices
  - [ ] Service URLs
  - [ ] Wake word settings
  - [ ] Architecture-specific settings
- [ ] Dependencies installed
- [ ] Orchestrator tested
- [ ] Logs verified

### Post-Deployment Verification
- [ ] Wake word detection working
- [ ] Speech-to-text functional
- [ ] Text-to-speech playback working
- [ ] Gateway communication successful
- [ ] No audio feedback loops
- [ ] Systemd service enabled (optional)

## Files to Sync

### Via Git (Code)
- `orchestrator/` (all Python modules)
- `*.sh` (all scripts)
- `requirements*.txt`
- `docker-compose.yml`
- `Dockerfile`
- `README.md`, `*.md` documentation
- `.env.example` (new comprehensive template)

### Via Rsync/SCP (Artifacts)
- `artifacts/precise-engine-armv7/precise-engine.tar.gz` (162M) - **Required for ARMv7**
- `docker/wakeword-models/hey-mycroft.pb` (26K)
- `docker/wakeword-models/hey-mycroft.pb.params` (132 bytes)

### Optional (Generated on Pi)
- `.venv_orchestrator/` - Created by install script
- `docker/silero-models/` - Auto-downloaded if needed
- `.env` - Generated from template during installation

## Next Steps

1. Create `.env.example` with comprehensive defaults from working Pi
2. Update `install_raspbian_remote.sh`:
   - Add architecture detection
   - Add artifact transfer logic
   - Add wake word configuration based on arch
   - Improve audio device detection
3. Create `sync_artifacts_to_pi.sh` helper script
4. Test deployment on new Pi with same hardware
5. Document any issues and update scripts
