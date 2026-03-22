# VNC Setup for Remmina — Integration Guide

## Overview

The VNC setup has been integrated into the OpenClaw Voice Orchestrator installation scripts to ensure that Raspberry Pi systems have Remmina-compatible remote desktop access immediately after installation.

## Files Created/Modified

### 1. **setup_vnc_remmina.sh** (NEW)
   - Standalone VNC setup script that configures wayvnc or x11vnc for Remmina compatibility
   - Can be run independently: `sudo bash setup_vnc_remmina.sh`
   - Supports:
     - `--status` — Check VNC service status
     - `--enable` — Enable and start VNC
     - `--disable` — Disable VNC services
   - **Location:** `/home/stever/projects/openclawstuff/openclaw-voice/setup_vnc_remmina.sh`

### 2. **install_raspbian.sh** (MODIFIED)
   - Added VNC setup step (Step 3b) after PulseAudio/Snapcast configuration
   - Runs during local Pi installation
   - Automatically calls `setup_vnc_remmina.sh`
   - **Location:** `/home/stever/projects/openclawstuff/openclaw-voice/install_raspbian.sh`

### 3. **install_raspbian_remote.sh** (MODIFIED)
   - Added VNC setup step (Step 5c) after environment configuration
   - Runs on remote Pi via SSH during remote installation
   - Automatically calls `setup_vnc_remmina.sh` on the target Pi
   - **Location:** `/home/stever/projects/openclawstuff/openclaw-voice/install_raspbian_remote.sh`

## VNC Configuration

After installation, the VNC server is configured as:

| Setting | Value |
|---------|-------|
| **Address** | 0.0.0.0 (all interfaces) |
| **Port** | 5900 |
| **Authentication** | None (no password) |
| **Security Type** | Type 1 (None) |
| **Remmina Compatible** | ✅ Yes |
| **Certificate Required** | ❌ No |
| **Auto-start** | ✅ Yes (systemd service) |

## Usage

### Option A: Local Installation
```bash
cd ~/openclaw-voice
sudo bash install_raspbian.sh
```
VNC will be set up automatically as part of the installation process.

### Option B: Remote Installation
```bash
cd ~/openclaw-voice
bash install_raspbian_remote.sh <pi_ip_address>
```
VNC will be set up automatically on the Pi.

### Option C: Manual VNC Setup (on existing Pi)
```bash
# Copy or download setup_vnc_remmina.sh to the Pi, then:
sudo bash setup_vnc_remmina.sh
```

## Connecting with Remmina

### On your computer (where you have Remmina installed):

1. **Open Remmina**
2. **Click "+" to create a new connection**
3. **Configure:**
   - **Name:** Pi VNC (or any name)
   - **Protocol:** VNC
   - **Server:** `<pi-ip-address>:5900` (or `<pi-hostname>.local:5900`)
   - **User name:** (leave blank)
   - **Password:** (leave blank)
   - **Security tab:**
     - Leave all certificate fields empty
     - Select "None" if prompted for security type

4. **Click "Connect"**

### Quick Connect String
If your Pi's IP is `10.1.1.72`:
```
remmina vnc://10.1.1.72:5900
```

## Verification

### Check VNC status on Pi:
```bash
sudo bash setup_vnc_remmina.sh --status
```

### View VNC logs:
```bash
sudo journalctl -u wayvnc -f    # For Wayland
# or
sudo journalctl -u x11vnc -f    # For X11
```

### Test VNC connectivity from another machine:
```bash
# From your local computer:
timeout 5 bash -lc '</dev/tcp/10.1.1.72/5900' && echo "VNC_OK" || echo "VNC_BLOCKED"
```

## Troubleshooting

### VNC not connecting
1. Verify the service is running:
   ```bash
   sudo systemctl status wayvnc
   ```

2. Check if port 5900 is listening:
   ```bash
   sudo ss -ltnp | grep 5900
   ```

3. Ensure firewall isn't blocking port 5900:
   ```bash
   sudo ufw allow 5900/tcp  # if ufw is enabled
   ```

### VNC connects but shows black screen
- Wait 2-3 seconds for the desktop to render
- Check if a compositor is running: `ps aux | grep -E "labwc|sway|gnome-shell"`
- Restart VNC: `sudo systemctl restart wayvnc`

### Certificate/TLS errors still appearing
- This indicates an old VNC config is still active
- Run: `sudo bash setup_vnc_remmina.sh` again to reconfigure
- Restart Remmina profile

### Remmina asks for password
- VNC should not require a password (auth is disabled)
- Just click "OK" with empty password fields
- Or update the connection profile in Remmina to clear saved credentials

## Architecture Detection

The `setup_vnc_remmina.sh` script automatically:
- Detects if the system is running Wayland or X11
- Installs `wayvnc` for Wayland-based systems (preferred for modern Pi OS)
- Falls back to `x11vnc` for X11 systems
- Configures the appropriate systemd service

## Security Considerations

⚠️ **Important:** This configuration disables VNC authentication for LAN access, which is:
- ✅ **Safe** for trusted local networks (your home/office)
- ❌ **Not recommended** for internet-facing systems

For internet access or untrusted networks:
1. Use SSH tunneling instead: `ssh -L 5902:127.0.0.1:5900 user@pi`
2. Or set up VNC password authentication (modify `/etc/wayvnc/config`)
3. Place a firewall/router between the Pi and the internet

## Reverting VNC Setup

To disable VNC:
```bash
sudo bash setup_vnc_remmina.sh --disable
```

To re-enable:
```bash
sudo bash setup_vnc_remmina.sh --enable
```
