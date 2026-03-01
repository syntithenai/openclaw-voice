# OpenClaw Voice Orchestrator - Raspbian Installation Guide

This guide covers installing the OpenClaw Voice Orchestrator on Raspbian (both 32-bit and 64-bit architectures).

## Quick Start

### One-Command Installation (Recommended)

```bash
cd /path/to/openclaw-voice-py
chmod +x install.sh
./install.sh
```

The installer will:
- ✅ Detect your Raspbian version and architecture
- ✅ Install all system dependencies
- ✅ Create a Python virtual environment
- ✅ Install all Python packages (core + optional)
- ✅ Guide you through configuration interactively
- ✅ Create helper scripts for easy activation and running

## Prerequisites

### Minimum Requirements
- **Raspbian OS**: Bullseye (or newer recommended)
- **Python**: 3.7+ (3.9+ recommended)
- **RAM**: 1GB minimum (2GB+ recommended for optional dependencies)
- **Disk Space**: 2GB+ for installation and models
- **Network**: Stable internet connection for package downloads

### Supported Architectures
- **32-bit**: ARMv6/ARMv7 (Raspberry Pi 3, 4, Zero 2W)
- **64-bit**: ARM64 (Raspberry Pi 4 with 64-bit OS, Pi 5)

### Raspbian Versions
- Bullseye (Debian 11)
- Bookworm (Debian 12)
- Buster (Debian 10) - not recommended, partial support

## Installation Steps

### Step 1: Prepare Your System

Connect to your Raspberry Pi via SSH or open a terminal:

```bash
# Update package lists
sudo apt-get update
sudo apt-get upgrade -y

# Verify Python 3 is available
python3 --version
pip3 --version
```

### Step 2: Clone/Navigate to Repository

```bash
cd /path/to/openclaw-voice-py
```

### Step 3: Run the Installer

```bash
chmod +x install.sh
./install.sh
```

The installer is interactive and will prompt you for:

#### Audio Configuration
- **Capture device**: Microphone input (default: "default")
- **Playback device**: Speaker output (default: "default")

To list available devices and test them:
```bash
python3 -m sounddevice
```

#### Gateway Configuration
- **Gateway URL**: WebSocket address of OpenClaw gateway
  - Example: `ws://openclaw.local:8000`
  - Example: `ws://192.168.1.100:8000`
- **Gateway token**: Authentication token for secure communication

#### STT Configuration (Whisper)
- **Whisper service URL**: HTTP address of running Whisper container
  - Default (local Docker): `http://localhost:10000`
  - Example (remote): `http://whisper.local:10000`

#### TTS Configuration (Piper)
- **Piper service URL**: HTTP address of running Piper container
  - Default (local Docker): `http://localhost:10001`
  - Example (remote): `http://piper.local:10001`
- **Voice**: Available voices in your Piper installation
  - Example: `en_US-amy-medium`, `en_US-lessac-medium`
- **Speed**: Speech rate multiplier (0.5 = slower, 1.5 = faster)

#### Wake Word Configuration
- **Model**: Which wake word to recognize
  - `hey_mycroft` (default, low false positives)
  - `hey_jarvis`
  - `timer`
  - `weather`
  - `alexa`
- **Confidence**: Detection threshold (0.0-1.0)
  - Higher = stricter (fewer false positives, but may miss real activations)
  - Default: 0.95

#### VAD Configuration
- **Backend**: Voice activity detector
  - `webrtc` (default, lightweight, recommended)
  - `silero` (optional, more accurate)
  - `none` (disable)
- **Aggressiveness**: Detection sensitivity (0-3)
  - 0 = least aggressive (may include background noise)
  - 3 = most aggressive (may cut off speech)

### Step 4: Verify Configuration

After installation, review the generated `.env` file:

```bash
nano .env
```

Make any adjustments to:
- Audio device names
- Service URLs and ports
- Wake word sensitivity
- Log levels

### Step 5: Test the Installation

Activate the virtual environment and test:

```bash
# Activate (automatic after install, or manually)
source .venv_orchestrator/bin/activate

# Test imports
python3 -c "import orchestrator; print('✓ Orchestrator imported successfully')"

# Test audio
python3 -m sounddevice  # List devices

# Test Whisper connection
curl -X GET http://localhost:10000/info

# Test Piper connection
curl -X GET http://localhost:10001/info
```

## Running the Orchestrator

### Option 1: Using the Run Script (Easiest)

```bash
./run.sh
```

### Option 2: Manual Activation

```bash
source .venv_orchestrator/bin/activate
python -m orchestrator.main
```

### Option 3: With Custom Configuration

```bash
source .venv_orchestrator/bin/activate
WHISPER_URL=http://remote-whisper:10000 python -m orchestrator.main
```

## Monitoring and Logs

### View Live Logs

```bash
tail -f orchestrator.log
```

### Monitor Specific Issues

```bash
# Watch for hotword detections
tail -f orchestrator.log | grep -i "wake word"

# Watch for transcription errors
tail -f orchestrator.log | grep -i "error\|warning"

# Watch for gateway communication
tail -f orchestrator.log | grep -i "gateway"
```

### Increase Log Detail

In `.env`:
```bash
LOG_LEVEL=DEBUG
```

## Troubleshooting

### Audio Not Captured ("No audio device found")

1. List available devices:
   ```bash
   python3 -m sounddevice
   ```

2. Update `.env` with correct device ID or name:
   ```bash
   AUDIO_CAPTURE_DEVICE=2  # Device ID
   # OR
   AUDIO_CAPTURE_DEVICE="USB Audio Device"  # Device name
   ```

3. Test audio capture:
   ```bash
   python3 -c "import sounddevice as sd; print(sd.rec(int(16000*2), samplerate=16000, channels=1))"
   ```

### Gateway Connection Failures

1. Verify gateway is running and accessible:
   ```bash
   curl -v ws://192.168.1.100:8000
   ```

2. Check `.env` for correct URL format:
   ```bash
   # Correct: ws:// (WebSocket), not http://
   GATEWAY_URL=ws://openclaw.local:8000
   ```

3. Verify token is correct:
   ```bash
   # Check OpenClaw server logs for expected token
   ```

4. Check network connectivity:
   ```bash
   ping openclaw.local
   # OR with IP:
   ping 192.168.1.100
   ```

### Whisper Service Unreachable

1. Verify Whisper container is running:
   ```bash
   docker ps | grep whisper
   ```

2. Check that port 10000 is open:
   ```bash
   curl -X GET http://localhost:10000/info
   ```

3. If using remote Whisper, verify network access:
   ```bash
   curl -X GET http://whisper.local:10000/info
   ```

### Piper Service Unreachable

1. Same as Whisper, but on port 10001:
   ```bash
   docker ps | grep piper
   curl -X GET http://localhost:10001/info
   ```

### High CPU Usage or Freezing

- Reduce `VAD_AGGRESSIVENESS` to `0` (less processing)
- Disable emotion detection (EMOTION_DETECTION_ENABLED=false)
- Use lighter wake word model (check available options)
- Monitor system resources:
  ```bash
  top
  # Press '1' to show per-core CPU usage
  ```

### "Wake word detected" but transcription wrong (e.g., "Hey, my craft" captured)

- The `WAKE_WORD_PREBUFFER_MS` has been reduced to 80ms (default) to minimize this
- Further reduce if needed:
  ```bash
  WAKE_WORD_PREBUFFER_MS=50
  ```
- Increase wake word confidence threshold:
  ```bash
  WAKE_WORD_CONFIDENCE=0.97  # Higher = stricter detection
  ```

### Memory Issues or Out of Memory Errors

1. Check available memory:
   ```bash
   free -h
   ```

2. Reduce running processes
3. Disable optional features:
   ```bash
   EMOTION_DETECTION_ENABLED=false
   VAD_BACKEND=webrtc  # Lighter than silero
   ```

4. Consider disabling Silero VAD in optional requirements:
   ```bash
   # Comment out in requirements-optional.txt:
   # silero-vad==5.1.2
   ```

## Advanced Configuration

### Custom Audio Frame Size

For lower latency or different VAD behavior:

```bash
# In .env
AUDIO_FRAME_MS=10   # 10ms frames (default 20ms)
```

### Emotion Detection (Optional)

Requires additional models (large download):

```bash
EMOTION_DETECTION_ENABLED=true
```

### Silero VAD (Optional, More Accurate)

```bash
VAD_BACKEND=silero
VAD_AGGRESSIVENESS=1  # 0-3, higher = stricter
```

### Remote Whisper/Piper

To use STT/TTS services running on different machines:

```bash
WHISPER_URL=http://whisper-server.local:10000
PIPER_URL=http://piper-server.local:10001
```

## Updating the Installation

### Update Python Packages

```bash
source .venv_orchestrator/bin/activate
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-optional.txt
```

### Update Orchestrator Code

If you've pulled the latest code:

```bash
source .venv_orchestrator/bin/activate
pip install --upgrade -r requirements.txt
# Then restart the orchestrator
./run.sh
```

## Uninstalling

To remove the installation while preserving `.env`:

```bash
# Remove virtual environment
rm -rf .venv_orchestrator

# Remove generated scripts (keep .env)
rm -f activate.sh run.sh

# Clean pip cache
pip cache purge
```

Full uninstall (including configuration):

```bash
rm -rf .venv_orchestrator activate.sh run.sh .env
```

## Performance Tips

### For Raspberry Pi Zero / Pi 3 (Limited Resources)

1. Disable optional features:
   ```bash
   EMOTION_DETECTION_ENABLED=false
   VAD_BACKEND=webrtc
   ```

2. Reduce TTS quality slightly:
   ```bash
   PIPER_VOICE=en_US-lessac-medium  # Smaller model
   ```

3. Run Whisper/Piper on a separate, more powerful machine

### For Raspberry Pi 4 / Pi 5 (Better Resources)

1. Use higher-quality voices:
   ```bash
   PIPER_VOICE=en_US-amy-medium
   ```

2. Enable optional features:
   ```bash
   VAD_BACKEND=silero
   EMOTION_DETECTION_ENABLED=true  # If resources permit
   ```

3. Run all services locally (Whisper, Piper, Orchestrator on same Pi)

## Getting Help

1. Check logs: `tail -f orchestrator.log`
2. Enable debug logging: Set `LOG_LEVEL=DEBUG` in `.env`
3. Verify all services are running: `docker ps`
4. Test network connectivity: `ping`, `curl` commands above
5. Check Raspberry Pi system logs: `journalctl -xe`

## See Also

- [README.md](README.md) - Main project documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands and troubleshooting
- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Development setup guide
