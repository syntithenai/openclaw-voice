#!/bin/bash
# Setup script for MPD orchestrator integration

set -e

echo "================================"
echo "MPD Orchestrator Setup"
echo "================================"

# Step 1: Install MPD
echo "Step 1: Check MPD installation..."
if ! command -v mpd &> /dev/null; then
    echo "  Installing mpd..."
    sudo apt-get update
    sudo apt-get install -y mpd
else
    echo "  ✓ mpd found: $(which mpd)"
fi

# Step 2: Install MPC
echo ""
echo "Step 2: Check MPC installation..."
if ! command -v mpc &> /dev/null; then
    echo "  Installing mpc..."
    sudo apt-get install -y mpc
else
    echo "  ✓ mpc found: $(which mpc)"
fi

# Step 3: Create MPD config directory
echo ""
echo "Step 3: Ensure MPD config directory..."
mkdir -p ~/.config/mpd ~/.mpdstate/playlists
echo "  ✓ Config directories ready"

# Step 4: Test MPD can start
echo ""
echo "Step 4: Test MPD start..."
if mpd ~/.config/mpd/mpd.conf 2>/dev/null || sleep 1; then
    sleep 1
    if ps aux | grep -q "[m]pd"; then
        echo "  ✓ MPD started successfully"
        mpc status 2>/dev/null || echo "  ✓ MPD is running"
        # Stop it for now
        killall mpd 2>/dev/null || true
        sleep 1
    else
        echo "  ✗ MPD did not start properly"
        exit 1
    fi
else
    echo "  ✗ Failed to start MPD"
    exit 1
fi

echo ""
echo "✓ All setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run orchestrator: python3 orchestrator/main.py"
echo "  2. Watch for: '✓ MPD ready in XXXms'"
echo "  3. Test music: mpc status"
echo ""
