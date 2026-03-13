#!/usr/bin/env bash
set -euo pipefail

RULE_PATH="/etc/udev/rules.d/99-openclaw-uinput.rules"
TARGET_GROUP="input"

if ! getent group "$TARGET_GROUP" >/dev/null 2>&1; then
  TARGET_GROUP="root"
  echo "WARN: group 'input' not found; falling back to group '$TARGET_GROUP'" >&2
fi

cat > "$RULE_PATH" <<EOF
KERNEL=="uinput", MODE="0660", GROUP="$TARGET_GROUP"
EOF

modprobe uinput || true
udevadm control --reload-rules
udevadm trigger /dev/uinput || true

# Apply immediately if node already exists
if [[ -e /dev/uinput ]]; then
  chgrp "$TARGET_GROUP" /dev/uinput || true
  chmod 660 /dev/uinput || true
fi

echo "uinput access configured"
ls -l /dev/uinput || true
