#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "ERROR: run as root (use sudo)."
  exit 1
fi

BT_USER="${OPENCLAW_BT_USER:-stever}"
BT_ALIAS="${OPENCLAW_BT_ALIAS:-Pi Two Bluetooth}"

if ! id "$BT_USER" >/dev/null 2>&1; then
  echo "ERROR: target user '$BT_USER' does not exist"
  exit 1
fi

BT_USER_UID="$(id -u "$BT_USER")"
BT_USER_HOME="$(getent passwd "$BT_USER" | cut -d: -f6)"

echo "==> Installing Bluetooth audio dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
  bluetooth \
  bluez \
  libspa-0.2-bluetooth \
  pulseaudio-utils \
  python3-dbus \
  python3-gi

echo "==> Configuring BlueZ defaults (/etc/bluetooth/main.conf)"
if [[ ! -f /etc/bluetooth/main.conf ]]; then
  touch /etc/bluetooth/main.conf
fi

upsert_conf_key() {
  local key="$1"
  local value="$2"
  if grep -qE "^[[:space:]]*${key}[[:space:]]*=" /etc/bluetooth/main.conf; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]*=.*|${key} = ${value}|" /etc/bluetooth/main.conf
  else
    printf '%s = %s\n' "$key" "$value" >> /etc/bluetooth/main.conf
  fi
}

upsert_conf_key "Class" "0x240414"
upsert_conf_key "DiscoverableTimeout" "0"
upsert_conf_key "PairableTimeout" "0"
upsert_conf_key "AlwaysPairable" "true"
upsert_conf_key "JustWorksRepairing" "always"

echo "==> Setting pretty hostname alias"
if [[ ! -f /etc/machine-info ]]; then
  touch /etc/machine-info
fi
if grep -q '^PRETTY_HOSTNAME=' /etc/machine-info; then
  sed -i -E "s|^PRETTY_HOSTNAME=.*|PRETTY_HOSTNAME=${BT_ALIAS}|" /etc/machine-info
else
  printf 'PRETTY_HOSTNAME=%s\n' "$BT_ALIAS" >> /etc/machine-info
fi

echo "==> Enforcing WirePlumber Bluetooth sink roles"
mkdir -p /etc/wireplumber/wireplumber.conf.d
cat > /etc/wireplumber/wireplumber.conf.d/51-openclaw-bluetooth.conf <<'EOF'
monitor.bluez.properties = {
  bluez5.roles = [ a2dp_sink hfp_hf hsp_hs ]
  bluez5.auto-connect = [ a2dp_sink hfp_hf hsp_hs ]
}
EOF

echo "==> Installing bluetooth host-mode helper"
cat > /usr/local/bin/bluetooth-host-mode.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail

BT_ALIAS="${BT_ALIAS}"
BT_ALIAS_ESCAPED="\${BT_ALIAS// /\\ }"

bluetoothctl <<BTEOF
power on
pairable on
discoverable on
system-alias \$BT_ALIAS_ESCAPED
quit
BTEOF
EOF
chmod +x /usr/local/bin/bluetooth-host-mode.sh

cat > /etc/systemd/system/bluetooth-host-mode.service <<'EOF'
[Unit]
Description=Bluetooth host mode defaults (OpenClaw)
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/bluetooth-host-mode.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

echo "==> Installing auto-pairing agent"
cat > /usr/local/bin/bluez-auto-agent.py <<'EOF'
#!/usr/bin/env python3
import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

AGENT_PATH = '/org/openclaw/bluez/auto_agent'


class Rejected(dbus.DBusException):
    _dbus_error_name = 'org.bluez.Error.Rejected'


class Agent(dbus.service.Object):
    @dbus.service.method('org.bluez.Agent1', in_signature='', out_signature='')
    def Release(self):
        pass

    @dbus.service.method('org.bluez.Agent1', in_signature='o', out_signature='s')
    def RequestPinCode(self, device):
        return '0000'

    @dbus.service.method('org.bluez.Agent1', in_signature='o', out_signature='u')
    def RequestPasskey(self, device):
        return dbus.UInt32(0)

    @dbus.service.method('org.bluez.Agent1', in_signature='ouq', out_signature='')
    def DisplayPasskey(self, device, passkey, entered):
        pass

    @dbus.service.method('org.bluez.Agent1', in_signature='os', out_signature='')
    def DisplayPinCode(self, device, pincode):
        pass

    @dbus.service.method('org.bluez.Agent1', in_signature='ou', out_signature='')
    def RequestConfirmation(self, device, passkey):
        return

    @dbus.service.method('org.bluez.Agent1', in_signature='o', out_signature='')
    def RequestAuthorization(self, device):
        return

    @dbus.service.method('org.bluez.Agent1', in_signature='os', out_signature='')
    def AuthorizeService(self, device, uuid):
        return

    @dbus.service.method('org.bluez.Agent1', in_signature='', out_signature='')
    def Cancel(self):
        pass


def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    agent = Agent(bus, AGENT_PATH)
    manager = dbus.Interface(bus.get_object('org.bluez', '/org/bluez'), 'org.bluez.AgentManager1')
    manager.RegisterAgent(AGENT_PATH, 'NoInputNoOutput')
    manager.RequestDefaultAgent(AGENT_PATH)
    GLib.MainLoop().run()


if __name__ == '__main__':
    main()
EOF
chmod +x /usr/local/bin/bluez-auto-agent.py

cat > /etc/systemd/system/bluez-auto-agent.service <<'EOF'
[Unit]
Description=BlueZ Auto Pairing Agent (NoInputNoOutput)
After=bluetooth.service
Requires=bluetooth.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/bin/bluez-auto-agent.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

echo "==> Installing Pulse/PipeWire sink auto-route hook"
cat > /usr/local/bin/openclaw-bt-route-sink.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if ! command -v pactl >/dev/null 2>&1; then
  exit 0
fi

connected_device="$(bluetoothctl devices Connected 2>/dev/null | awk '/^Device / {print $2; exit}')"
if [[ -z "$connected_device" ]]; then
  exit 0
fi

target_sink="$(pactl list short sinks 2>/dev/null | awk '{print $2}' | grep -E '^(bluez_output|bluez_sink)\.' | head -n1 || true)"
if [[ -z "$target_sink" ]]; then
  exit 0
fi

current_sink="$(pactl get-default-sink 2>/dev/null || true)"
if [[ "$current_sink" != "$target_sink" ]]; then
  pactl set-default-sink "$target_sink" >/dev/null 2>&1 || true
fi

pactl set-sink-mute "$target_sink" 0 >/dev/null 2>&1 || true
pactl set-sink-volume "$target_sink" 100% >/dev/null 2>&1 || true

while read -r sink_input_id _; do
  [[ -n "${sink_input_id:-}" ]] || continue
  pactl move-sink-input "$sink_input_id" "$target_sink" >/dev/null 2>&1 || true
done < <(pactl list short sink-inputs 2>/dev/null || true)
EOF
chmod +x /usr/local/bin/openclaw-bt-route-sink.sh

install -d -m 0755 "$BT_USER_HOME/.config/systemd/user"
cat > "$BT_USER_HOME/.config/systemd/user/openclaw-bt-route.service" <<'EOF'
[Unit]
Description=OpenClaw Bluetooth sink auto-route

[Service]
Type=oneshot
ExecStart=/usr/local/bin/openclaw-bt-route-sink.sh
EOF

cat > "$BT_USER_HOME/.config/systemd/user/openclaw-bt-route.timer" <<'EOF'
[Unit]
Description=Run OpenClaw Bluetooth sink auto-route periodically

[Timer]
OnBootSec=30
OnUnitActiveSec=10
AccuracySec=2
Unit=openclaw-bt-route.service

[Install]
WantedBy=timers.target
EOF

chown -R "$BT_USER":"$BT_USER" "$BT_USER_HOME/.config/systemd/user"

echo "==> Enabling linger and user timer for $BT_USER"
loginctl enable-linger "$BT_USER" >/dev/null 2>&1 || true
sudo -u "$BT_USER" XDG_RUNTIME_DIR="/run/user/$BT_USER_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$BT_USER_UID/bus" systemctl --user daemon-reload || true
sudo -u "$BT_USER" XDG_RUNTIME_DIR="/run/user/$BT_USER_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$BT_USER_UID/bus" systemctl --user enable --now openclaw-bt-route.timer || true

echo "==> Enabling Bluetooth services"
systemctl daemon-reload
systemctl enable --now bluetooth.service
systemctl enable --now bluetooth-host-mode.service
systemctl enable --now bluez-auto-agent.service
systemctl restart bluetooth.service
systemctl restart bluez-auto-agent.service
systemctl restart bluetooth-host-mode.service || true

echo "==> Bluetooth audio host setup complete"
echo "    alias: $BT_ALIAS"
echo "    user sink auto-route timer: openclaw-bt-route.timer"