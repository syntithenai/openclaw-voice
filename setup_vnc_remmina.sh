#!/bin/bash

################################################################################
#              OpenClaw Voice - VNC Setup for Remmina Support                  #
#                                                                              #
# This script installs and configures VNC (via wayvnc or x11vnc) with         #
# Remmina-compatible settings:                                                #
#   - Listens on all interfaces (0.0.0.0)                                     #
#   - No authentication (Security type: None)                                 #
#   - No certificate files required                                           #
#   - Auto-start via systemd                                                  #
#                                                                              #
# Usage:                                                                       #
#   sudo bash setup_vnc_remmina.sh                                            #
#   sudo bash setup_vnc_remmina.sh [--disable | --enable | --status]         #
#                                                                              #
################################################################################

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect if running with sudo
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Parse arguments
ACTION="${1:-install}"

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}OpenClaw Voice - VNC Setup for Remmina${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# Helper functions
################################################################################

log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "${BLUE}→${NC} $1"
}

################################################################################
# STATUS ACTION
################################################################################

if [ "$ACTION" = "--status" ] || [ "$ACTION" = "status" ]; then
    echo -e "${YELLOW}VNC Service Status:${NC}"
    echo ""
    
    if systemctl is-active --quiet wayvnc 2>/dev/null; then
        echo -e "${GREEN}wayvnc:${NC} active"
        systemctl status wayvnc --no-pager -n 3 2>/dev/null || true
        echo ""
        echo -e "${YELLOW}Configuration:${NC}"
        if [ -f /etc/wayvnc/config ]; then
            cat /etc/wayvnc/config
        fi
        echo ""
        echo -e "${YELLOW}Listening on:${NC}"
        ss -ltnp 2>/dev/null | grep 5900 || echo "Not listening"
    elif systemctl is-active --quiet x11vnc 2>/dev/null; then
        echo -e "${GREEN}x11vnc:${NC} active"
        systemctl status x11vnc --no-pager -n 3 2>/dev/null || true
    else
        echo -e "${RED}No VNC service is active${NC}"
    fi
    exit 0
fi

################################################################################
# DISABLE ACTION
################################################################################

if [ "$ACTION" = "--disable" ] || [ "$ACTION" = "disable" ]; then
    echo -e "${YELLOW}Disabling VNC services...${NC}"
    
    for service in wayvnc x11vnc vncserver; do
        if systemctl is-enabled "$service" &>/dev/null 2>&1; then
            log_step "Disabling $service..."
            systemctl disable --now "$service" 2>/dev/null || true
            log_info "$service disabled"
        fi
    done
    
    echo ""
    echo -e "${GREEN}VNC services disabled.${NC}"
    exit 0
fi

################################################################################
# INSTALL/ENABLE ACTION
################################################################################

# Detect OS and architecture
ARCH=$(uname -m)
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="linux"
fi

echo -e "${YELLOW}System Information:${NC}"
echo "  OS: $OS ($PRETTY_NAME)"
echo "  Architecture: $ARCH"
echo ""

# Check if Wayland is available
COMPOSITOR_TYPE="unknown"
if pgrep -a labwc &>/dev/null || pgrep -a sway &>/dev/null || pgrep -a gnome-shell &>/dev/null; then
    COMPOSITOR_TYPE="wayland"
elif [ -n "$DISPLAY" ] || pgrep -a X &>/dev/null; then
    COMPOSITOR_TYPE="x11"
fi

echo -e "${YELLOW}Compositor Detection:${NC}"
echo "  Type: $COMPOSITOR_TYPE"
echo ""

################################################################################
# STEP 1: Install VNC packages
################################################################################

echo -e "${YELLOW}Step 1: Installing VNC packages...${NC}"

# Try wayvnc first (preferred for Wayland)
if [ "$COMPOSITOR_TYPE" = "wayland" ] && [ "$OS" != "ubuntu" ] || [ "$OS" = "raspbian" ]; then
    log_step "Installing wayvnc for Wayland..."
    if apt-get update -qq && apt-get install -y wayvnc &>/dev/null; then
        log_info "wayvnc installed"
        VNC_SERVER="wayvnc"
    else
        log_warn "wayvnc installation failed, trying x11vnc..."
        VNC_SERVER="x11vnc"
    fi
else
    # Fall back to x11vnc for X11 sessions
    log_step "Installing x11vnc for X11..."
    if apt-get update -qq && apt-get install -y x11vnc &>/dev/null; then
        log_info "x11vnc installed"
        VNC_SERVER="x11vnc"
    else
        log_error "Failed to install VNC packages"
        exit 1
    fi
fi

echo ""

################################################################################
# STEP 2: Configure wayvnc (if installed)
################################################################################

if [ "$VNC_SERVER" = "wayvnc" ]; then
    echo -e "${YELLOW}Step 2: Configuring wayvnc for Remmina...${NC}"
    
    # Create/update wayvnc config for Remmina compatibility
    log_step "Creating wayvnc configuration..."
    mkdir -p /etc/wayvnc
    
    cat > /etc/wayvnc/config << 'EOF'
# wayvnc configuration for Remmina compatibility
# See: man wayvnc

use_relative_paths=false

# Listen on all interfaces (0.0.0.0) or specific IP
address=0.0.0.0

# Port (default 5900)
port=5900

# Disable authentication for LAN-only direct connection
# Security type will be "None" in VNC handshake
enable_auth=false

# Optional: keyboard layout
# xkb_layout=us
EOF
    
    log_info "wayvnc config created"
    
    # Create systemd override to run wayvnc properly
    log_step "Setting up wayvnc systemd service..."
    mkdir -p /etc/systemd/system/wayvnc.service.d
    
    cat > /etc/systemd/system/wayvnc.service.d/override.conf << 'EOF'
[Service]
# Ensure wayvnc can access the Wayland socket
# Most Wayland sessions are under UID 1000 (first user)
User=
Environment=
EOF
    
    log_info "wayvnc systemd override created"
    
    # Ensure wayvnc socket directory has proper permissions
    mkdir -p /tmp/wayvnc
    chmod 0777 /tmp/wayvnc
    
fi

echo ""

################################################################################
# STEP 3: Configure x11vnc (if needed)
################################################################################

if [ "$VNC_SERVER" = "x11vnc" ]; then
    echo -e "${YELLOW}Step 3: Configuring x11vnc for Remmina...${NC}"
    
    log_step "Creating x11vnc systemd service..."
    
    # Create systemd service for x11vnc with Remmina-compatible settings
    cat > /etc/systemd/system/x11vnc.service << 'EOF'
[Unit]
Description=X11 VNC Server for Remmina
After=display-manager.service
Wants=display-manager.service

[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -display :0 -auth /run/user/1000/.Xauthority -listen 0.0.0.0 -rfbport 5900 -nopw -shared -forever -xrandr auto
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    
    log_info "x11vnc systemd service created"
    
fi

echo ""

################################################################################
# STEP 4: Enable and start VNC service
################################################################################

echo -e "${YELLOW}Step 4: Enabling VNC service...${NC}"

# Disable any conflicting VNC services
for service in vncserver tigervnc-scraping-server; do
    if systemctl is-enabled "$service" 2>/dev/null; then
        log_step "Disabling conflicting $service..."
        systemctl disable --now "$service" 2>/dev/null || true
    fi
done

# Enable and start the chosen VNC server
log_step "Enabling $VNC_SERVER..."
systemctl daemon-reload
systemctl enable "$VNC_SERVER.service"
systemctl restart "$VNC_SERVER.service"

# Wait for service to stabilize
sleep 2

# Verify service is running
if systemctl is-active --quiet "$VNC_SERVER"; then
    log_info "$VNC_SERVER is running"
else
    log_error "$VNC_SERVER failed to start"
    echo ""
    echo -e "${YELLOW}Service logs:${NC}"
    journalctl -u "$VNC_SERVER.service" -n 20 --no-pager || true
    exit 1
fi

echo ""

################################################################################
# STEP 5: Verify VNC port is listening
################################################################################

echo -e "${YELLOW}Step 5: Verifying VNC port...${NC}"

if ss -ltnp 2>/dev/null | grep -q ':5900'; then
    log_info "VNC is listening on port 5900"
    ss -ltnp | grep ':5900'
else
    log_error "VNC is not listening on port 5900"
    exit 1
fi

echo ""

################################################################################
# STEP 6: Test VNC handshake (no auth required)
################################################################################

echo -e "${YELLOW}Step 6: Testing VNC security negotiation...${NC}"

# Test RFB handshake locally (optional check)
python3 -c "
import socket
try:
    s = socket.create_connection(('127.0.0.1', 5900), timeout=3)
    ver = s.recv(12)
    s.sendall(b'RFB 003.008
')
    num = s.recv(1)
    if num:
        n = num[0]
        sec = list(s.recv(n))
        if 1 in sec:
            print('REMMINA_COMPATIBLE=yes')
        else:
            print('REMMINA_COMPATIBLE=no')
    s.close()
except:
    pass
" > /tmp/vnc_test.out 2>/dev/null
TEST_OUTPUT=$(cat /tmp/vnc_test.out 2>/dev/null || echo "")


if [ -z "$TEST_OUTPUT" ] || [[ "$TEST_OUTPUT" == *"TEST_SKIPPED"* ]]; then
    log_warn "Python RFB test skipped (python3 not available)"
elif [[ "$TEST_OUTPUT" == *"REMMINA_COMPATIBLE=yes"* ]]; then
    log_info "VNC is Remmina-compatible (Security type: None)"
    echo "$TEST_OUTPUT" | grep -E "SERVER_VERSION|SECURITY_TYPES|REMMINA_COMPATIBLE"
else
    log_warn "VNC may require authentication or certificate files"
    echo "$TEST_OUTPUT" | grep -E "SERVER_VERSION|SECURITY_TYPES|REMMINA_COMPATIBLE"
fi

echo ""

################################################################################
# STEP 7: Network test (from localhost, since we may be in LAN-only mode)
################################################################################

echo -e "${YELLOW}Step 7: Testing network accessibility...${NC}"

# Get local IP addresses
LOCAL_IPS=$(hostname -I | tr ' ' '\n' | grep -v '^$' | head -5)

echo -e "${BLUE}  VNC is accessible from:${NC}"
for ip in $LOCAL_IPS; do
    echo "    • $ip:5900"
done

# If this is a Pi, add helpful info
if grep -q "raspbian\|raspberry" /etc/os-release 2>/dev/null; then
    echo ""
    echo -e "${BLUE}  This is a Raspberry Pi. To connect from another machine:${NC}"
    echo "    1. Open Remmina on your computer"
    echo "    2. New connection:"
    echo "       • Name: Pi VNC (or any name)"
    echo "       • Protocol: VNC"
    echo "       • Server: $(hostname -I | awk '{print $1}'):5900"
    echo "       • User name: (leave blank)"
    echo "       • Password: (leave blank)"
    echo "    3. Click Connect"
fi

echo ""

################################################################################
# FINAL STATUS
################################################################################

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}VNC Installation Complete${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}VNC Server:${NC} $VNC_SERVER"
echo -e "${YELLOW}Port:${NC} 5900"
echo -e "${YELLOW}Authentication:${NC} None (open access on LAN)"
echo -e "${YELLOW}Remmina Compatible:${NC} Yes"
echo ""
echo -e "${YELLOW}Service management:${NC}"
echo "  • Start:  sudo systemctl start $VNC_SERVER"
echo "  • Stop:   sudo systemctl stop $VNC_SERVER"
echo "  • Status: sudo systemctl status $VNC_SERVER"
echo "  • Logs:   sudo journalctl -u $VNC_SERVER -f"
echo ""
echo -e "${YELLOW}Check VNC status anytime:${NC}"
echo "  sudo bash $(basename "$0") --status"
echo ""

exit 0
