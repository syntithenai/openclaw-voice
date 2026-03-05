#!/usr/bin/env bash
# Sync all essential artifacts to a new Raspberry Pi
# Usage: ./sync_artifacts_to_pi.sh <ssh_target> [architecture]
#
# Arguments:
#   ssh_target: SSH connection string (e.g., stever@10.1.1.211 or pi_new)
#   architecture: Optional. armv7, arm64, or auto (default: auto-detect)
#
# This script copies all large files not in git that are essential for operation:
# - Precise engine binary (architecture-specific)
# - Wake word models
# - Any pre-downloaded models

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SSH_TARGET="${1:-}"
ARCH="${2:-auto}"

if [ -z "$SSH_TARGET" ]; then
    echo -e "${RED}Error: SSH target required${NC}"
    echo "Usage: $0 <ssh_target> [architecture]"
    echo "Example: $0 stever@10.1.1.211"
    echo "Example: $0 pi_new armv7"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}OpenClaw Voice - Artifact Sync${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Target: $SSH_TARGET"
echo ""

# Detect architecture if auto
if [ "$ARCH" = "auto" ]; then
    echo -e "${YELLOW}Detecting target architecture...${NC}"
    DETECTED_ARCH=$(ssh "$SSH_TARGET" "uname -m" 2>/dev/null || echo "unknown")
    case "$DETECTED_ARCH" in
        armv7l|armv6l)
            ARCH="armv7"
            ;;
        aarch64|armv8*)
            ARCH="arm64"
            ;;
        x86_64)
            ARCH="x86_64"
            ;;
        *)
            echo -e "${RED}Error: Unknown architecture: $DETECTED_ARCH${NC}"
            exit 1
            ;;
    esac
    echo -e "${GREEN}  ✓ Detected: $DETECTED_ARCH → using $ARCH artifacts${NC}"
fi

# Create remote directory structure
echo ""
echo -e "${YELLOW}Creating directory structure on target...${NC}"
ssh "$SSH_TARGET" "mkdir -p ~/openclaw-voice/docker/wakeword-models" 2>/dev/null || true
echo -e "${GREEN}  ✓ Directories created${NC}"

# Sync wake word models (needed for Precise on armv7)
if [ -d "./docker/wakeword-models" ]; then
    echo ""
    echo -e "${YELLOW}Syncing wake word models...${NC}"
    rsync -avz --progress \
        ./docker/wakeword-models/ \
        "$SSH_TARGET:~/openclaw-voice/docker/wakeword-models/" || {
        echo -e "${YELLOW}  ⚠ Wake word model sync failed (may not exist locally)${NC}"
    }
    echo -e "${GREEN}  ✓ Wake word models synced${NC}"
else
    echo -e "${YELLOW}  ⚠ No local wake word models found (./docker/wakeword-models)${NC}"
fi

# Sync Precise engine for ARMv7
if [ "$ARCH" = "armv7" ]; then
    PRECISE_TAR="./artifacts/precise-engine-armv7/precise-engine.tar.gz"
    if [ -f "$PRECISE_TAR" ]; then
        echo ""
        echo -e "${YELLOW}Syncing Precise engine for ARMv7...${NC}"
        FILE_SIZE=$(ls -lh "$PRECISE_TAR" | awk '{print $5}')
        echo "  File: $PRECISE_TAR ($FILE_SIZE)"
        
        rsync -avz --progress \
            "$PRECISE_TAR" \
            "$SSH_TARGET:~/openclaw-voice/precise-engine.tar.gz" || {
            echo -e "${RED}  ✗ Failed to sync Precise engine${NC}"
            exit 1
        }
        echo -e "${GREEN}  ✓ Precise engine synced${NC}"
        
        # Extract on remote
        echo -e "${YELLOW}Extracting Precise engine on target...${NC}"
        ssh "$SSH_TARGET" << 'EXTRACT_CMD'
cd ~/openclaw-voice
if [ -f precise-engine.tar.gz ]; then
    rm -rf precise-engine
    tar -xzf precise-engine.tar.gz
    if [ -x "precise-engine/precise-engine" ]; then
        echo "  ✓ Precise engine extracted successfully"
        ls -lh precise-engine/precise-engine
    else
        echo "  ✗ ERROR: precise-engine binary not found after extraction"
        exit 1
    fi
else
    echo "  ✗ ERROR: precise-engine.tar.gz not found"
    exit 1
fi
EXTRACT_CMD
        echo -e "${GREEN}  ✓ Precise engine extracted${NC}"
    else
        echo -e "${RED}  ✗ ERROR: Precise engine not found: $PRECISE_TAR${NC}"
        echo -e "${YELLOW}  → Build it first with: ./build_precise_engine_armv7.sh${NC}"
        exit 1
    fi
fi

# Sync Precise engine for ARM64 (if available)
if [ "$ARCH" = "arm64" ]; then
    PRECISE_TAR="./artifacts/precise-engine-arm64/precise-engine.tar.gz"
    if [ -f "$PRECISE_TAR" ]; then
        echo ""
        echo -e "${YELLOW}Syncing Precise engine for ARM64...${NC}"
        FILE_SIZE=$(ls -lh "$PRECISE_TAR" | awk '{print $5}')
        echo "  File: $PRECISE_TAR ($FILE_SIZE)"
        
        rsync -avz --progress \
            "$PRECISE_TAR" \
            "$SSH_TARGET:~/openclaw-voice/precise-engine.tar.gz" || {
            echo -e "${YELLOW}  ⚠ Failed to sync Precise engine (OpenWakeWord can be used instead)${NC}"
        }
        
        # Extract on remote
        ssh "$SSH_TARGET" << 'EXTRACT_CMD'
cd ~/openclaw-voice
if [ -f precise-engine.tar.gz ]; then
    rm -rf precise-engine
    tar -xzf precise-engine.tar.gz
    if [ -x "precise-engine/precise-engine" ]; then
        echo "  ✓ Precise engine extracted successfully"
    fi
fi
EXTRACT_CMD
    else
        echo -e "${YELLOW}  ℹ No Precise engine for ARM64 (will use OpenWakeWord)${NC}"
    fi
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Artifact Sync Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Synced artifacts:"
if [ "$ARCH" = "armv7" ]; then
    echo "  ✓ Precise engine for ARMv7 (required)"
    echo "  ✓ hey-mycroft.pb wake word model"
elif [ "$ARCH" = "arm64" ]; then
    echo "  ✓ Wake word models"
    echo "  ℹ OpenWakeWord will be used (no binary needed)"
fi
echo ""
echo "Next steps:"
echo "1. Ensure code is synced: ssh $SSH_TARGET 'cd ~/openclaw-voice && git pull'"
echo "2. Run install script: ssh $SSH_TARGET 'cd ~/openclaw-voice && bash install_raspbian.sh'"
echo "3. Configure .env with correct service URLs and audio devices"
echo ""
