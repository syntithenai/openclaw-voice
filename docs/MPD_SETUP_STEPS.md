# MPD Setup Verification

## Quick Setup

Since `mpc` is not installed, run these commands manually in your terminal:

```bash
# Install mpc (MPD command-line client)
sudo apt-get install -y mpc

# Create MPD config directory if it doesn't exist
mkdir -p ~/.config/mpd ~/.mpdstate/playlists

# Create basic MPD config (if needed)
cat > ~/.config/mpd/mpd.conf << 'EOF'
music_directory "~/Music"
playlist_directory "~/.config/mpd/playlists"
db_file "~/.config/mpd/database"
log_file "~/.config/mpd/mpd.log"
pid_file "~/.config/mpd/mpd.pid"
state_file "~/.config/mpd/state"

bind_to_address "127.0.0.1"
port "6600"

audio_output {
    type "pipewire"
    name "PipeWire Output"
}

audio_output {
    type "alsa"
    name "ALSA Output"
}

log_level "default"
auto_update "yes"
restore_paused "yes"
EOF

# Test that mpc can connect
mpc status

# You should see output like:
# (nothing playing) or the status of any playing track
```

## If `mpc status` shows "MPD_HOST environment variable is not set"

This means MPD is not running. Normally the orchestrator starts it for you. For manual debugging only:

```bash
mpd ~/.config/mpd/mpd.conf
```

Then test again:
```bash
mpc status
```

## Testing the Orchestrator Integration

Once `mpc status` works without errors, you can:

```bash
# Run the orchestrator
cd /home/stever/projects/openclawstuff/openclaw-voice
python3 orchestrator/main.py

# In another terminal, verify MPD is still running:
mpc status
```

## Troubleshooting

### "Cannot stat /root/.config/mpd/mpd.conf" or permission errors
```bash
mkdir -p ~/.config/mpd
chmod 700 ~/.config/mpd
```

### "Connection refused" on port 6600
```bash
# Check if MPD is running:
ps aux | grep mpd

# Check if port is in use:
lsof -i :6600

# Start MPD:
mpd ~/.config/mpd/mpd.conf
```

### "Permission denied" creating state file
```bash
chmod 700 ~/.mpdstate
```

## Success Indicators

✓ `mpc status` shows output (not an error)
✓ `ps aux | grep mpd` shows an mpd process
✓ Orchestrator logs show "✓ MPD ready in XXXms"

---

**Next**: Run setup commands above, then start the orchestrator with `python3 orchestrator/main.py`
