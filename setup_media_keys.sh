#!/bin/bash
# Quick setup script for media key detection

echo "===================================================================="
echo "Media Key Detection Setup for Anker Conference Speaker"
echo "===================================================================="
echo ""

# Check if evdev is installed
if ! python3 -c "import evdev" 2>/dev/null; then
    echo "❌ evdev library not found"
    echo "   Installing evdev..."
    pip3 install evdev
    if [ $? -ne 0 ]; then
        echo "   ❌ Failed to install evdev"
        exit 1
    fi
    echo "   ✅ evdev installed"
else
    echo "✅ evdev library already installed"
fi

echo ""
echo "--------------------------------------------------------------------"
echo "Step 1: Check Current Permissions"
echo "--------------------------------------------------------------------"

if groups | grep -q "\binput\b"; then
    echo "✅ You are already in the 'input' group"
    HAS_PERMISSION=1
else
    echo "❌ You are NOT in the 'input' group"
    echo ""
    echo "To add yourself to the input group, run:"
    echo "   sudo usermod -a -G input $USER"
    echo ""
    echo "⚠️  You will need to log out and back in for this to take effect!"
    echo ""
    read -p "Add yourself to input group now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo usermod -a -G input $USER
        echo "✅ Added to input group"
        echo "⚠️  LOG OUT AND BACK IN for changes to take effect!"
        HAS_PERMISSION=0
    else
        echo "Skipping... You can add the group later."
        HAS_PERMISSION=0
    fi
fi

echo ""
echo "--------------------------------------------------------------------"
echo "Step 2: Scan for Media Devices"
echo "--------------------------------------------------------------------"
echo ""

if [ $HAS_PERMISSION -eq 1 ]; then
    echo "Scanning for devices with media keys..."
    python3 identify_speaker.py
else
    echo "Scanning for devices with media keys (using sudo)..."
    sudo python3 identify_speaker.py
fi

echo ""
echo "--------------------------------------------------------------------"
echo "Step 3: Test Button Detection"
echo "--------------------------------------------------------------------"
echo ""
echo "Press buttons on your Anker speaker to test detection."
echo ""

if [ $HAS_PERMISSION -eq 1 ]; then
    echo "Starting test... (Press Ctrl+C to stop)"
    echo ""
    python3 find_anker_device.py
else
    echo "Starting test with sudo... (Press Ctrl+C to stop)"
    echo ""
    sudo python3 find_anker_device.py
fi

echo ""
echo "===================================================================="
echo "Setup Complete!"
echo "===================================================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and set:"
echo "     MEDIA_KEYS_ENABLED=true"
echo "     MEDIA_KEYS_DEVICE_FILTER=Burr-Brown  # Or your device name"
echo ""
echo "  2. Restart the orchestrator:"
echo "     ./run_voice_demo.sh"
echo ""
echo "  3. Press buttons on your Anker speaker to control music!"
echo ""
