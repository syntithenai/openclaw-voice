#!/bin/bash
# OpenClaw Voice Orchestrator - Quick Start Guide
# This file documents the fastest way to get up and running

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════╗
║   OpenClaw Voice Orchestrator - Quick Start                       ║
║   (Raspbian 32-bit & 64-bit)                                      ║
╚════════════════════════════════════════════════════════════════════╝

[1] ONE-COMMAND INSTALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    cd /path/to/openclaw-voice-py
    bash install.sh

Then answer the interactive prompts:
  • Audio devices (press Enter for defaults)
  • Gateway URL: ws://openclaw.local:8000
  • Gateway token: (your token)
  • Whisper URL: http://localhost:10000
  • Piper URL: http://localhost:10001
  • Wake word: hey_mycroft (default recommended)
  • Log level: INFO

[2] QUICK VERIFY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    source .venv_orchestrator/bin/activate
    python3 -c "import orchestrator; print('✓ OK')"

[3] RUNNING THE ORCHESTRATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    bash run.sh

View logs in another terminal:

    tail -f orchestrator.log

[4] TROUBLESHOOTING QUICK CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ☐ Audio capture working?
      python3 -m sounddevice

  ☐ Gateway accessible?
      curl -v ws://openclaw.local:8000

  ☐ Whisper running?
      curl http://localhost:10000/info

  ☐ Piper running?
      curl http://localhost:10001/info

  ☐ Check .env configuration?
      nano .env

[5] COMMON CONFIGURATION OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Edit .env for:

  • Microphone not detected?
      AUDIO_CAPTURE_DEVICE=2  (use device ID from sounddevice)

  • Wrong audio output?
      AUDIO_PLAYBACK_DEVICE=1

  • Too sensitive to wake word?
      WAKE_WORD_CONFIDENCE=0.97

  • Capturing wake word in transcript?
      WAKE_WORD_PREBUFFER_MS=50  (reduced from 80)

  • Low RAM / freezing?
      EMOTION_DETECTION_ENABLED=false
      VAD_BACKEND=webrtc

  • Want detailed troubleshooting?
      LOG_LEVEL=DEBUG

[6] SYSTEM REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Raspbian (Bullseye or newer recommended)
  • Python 3.7+ (3.9+ better)
  • 1GB RAM minimum, 2GB+ recommended
  • 2GB disk space minimum
  • Stable internet connection

[7] ARCHITECTURE SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  32-bit (ARMv7):   ✓ Supported
  64-bit (ARM64):   ✓ Supported

  The installer auto-detects and optimizes for your platform.

[8] DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Full install guide:     RASPBIAN_INSTALL.md
  Quick reference:        QUICK_REFERENCE.md
  Configuration docs:     ENVIRONMENT_SETUP.md
  Troubleshooting:        See RASPBIAN_INSTALL.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready? Run:

    bash install.sh

Questions? Check RASPBIAN_INSTALL.md for detailed help.

EOF
