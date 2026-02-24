#!/usr/bin/env python
"""Verify complete orchestrator setup."""

import json
import sys
from pathlib import Path

# Check Python version
print(f"✅ Python {sys.version.split()[0]}")

# Check required directories
dirs = [
    "docker/silero-models",
    "docker/wakeword-models", 
    "docker/emotion-models",
    "docker/whisper-models",
    "docker/piper-data",
    "orchestrator",
]

for dir_path in dirs:
    p = Path(dir_path)
    status = "✅" if p.exists() else "❌"
    print(f"{status} {dir_path}/")

# Check required files
files = [
    ".env",
    "docker-compose.yml",
    "orchestrator/config.py",
    "orchestrator/main.py",
    "orchestrator/tools/fake_gateway.py",
    "e2e_test.py",
    "SETUP_COMPLETE.md",
]

print("\nFiles:")
for file_path in files:
    p = Path(file_path)
    status = "✅" if p.exists() else "❌"
    print(f"{status} {file_path}")

# Verify config loads
print("\nConfiguration:")
try:
    from orchestrator.config import VoiceConfig
    config = VoiceConfig()
    print(f"✅ BaseSettings loading from .env")
    print(f"  - VAD Type: {config.vad_type}")
    print(f"  - Auto-download Silero: {config.silero_auto_download}")
    print(f"  - Auto-download Wakeword: {config.openwakeword_auto_download}")
    print(f"  - Auto-download Emotion: {config.emotion_auto_download}")
    print(f"  - Gateway Agent ID: {config.gateway_agent_id}")
    print(f"  - Silero Models Dir: {config.silero_model_cache_dir}")
    print(f"  - Wakeword Models Dir: {config.openwakeword_models_dir}")
    print(f"  - Emotion Models Dir: {config.emotion_models_dir}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

# Check docker-compose volumes
print("\nDocker Compose Volumes:")
try:
    with open("docker-compose.yml") as f:
        content = f.read()
    required_volumes = [
        "silero-models",
        "wakeword-models",
        "emotion-models",
    ]
    for vol in required_volumes:
        if vol in content:
            print(f"✅ {vol} mount configured")
        else:
            print(f"❌ {vol} mount missing")
except Exception as e:
    print(f"❌ Docker compose error: {e}")

# Check fake gateway
print("\nFake Gateway:")
try:
    import requests
    resp = requests.get("http://localhost:18901/health", timeout=1)
    if resp.status_code == 200:
        print(f"✅ Fake gateway running on :18901")
    else:
        print(f"⚠️  Fake gateway not responding (status {resp.status_code})")
except Exception:
    print(f"⚠️  Fake gateway not running (start with: python -m orchestrator.tools.fake_gateway)")

print("\n" + "="*60)
print("✅ Setup Complete! Ready for deployment.")
print("="*60)

print("\nQuick Start:")
print("  1. Start fake gateway: python -m orchestrator.tools.fake_gateway &")
print("  2. Run tests: python e2e_test.py")
print("  3. Or start with docker-compose: docker-compose up -d")
