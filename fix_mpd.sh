#!/bin/bash
# MPD Setup Script

set -e

echo "Setting up MPD directories..."

# Create all required directories
mkdir -p ~/.config/mpd/playlists
mkdir -p ~/.mpdstate/playlists

# Create empty database file
touch ~/.config/mpd/database

echo "✓ Directories created"
ls -la ~/.config/mpd/

echo ""
echo "Directories ready. Now starting MPD..."
echo "Run: mpd ~/.config/mpd/mpd.conf"
echo ""
echo "Then verify with: mpc status"
