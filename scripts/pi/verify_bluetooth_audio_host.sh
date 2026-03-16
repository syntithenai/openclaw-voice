#!/usr/bin/env bash

set -euo pipefail

BT_USER="${OPENCLAW_BT_USER:-stever}"
EXPECTED_ALIAS="${OPENCLAW_BT_ALIAS:-Pi Two Bluetooth}"
AUTO_FIX_RAW="${OPENCLAW_BT_AUTOFIX:-true}"
SETUP_SCRIPT_DEFAULT="/home/${BT_USER}/openclaw-voice/scripts/pi/setup_bluetooth_audio_host.sh"
SETUP_SCRIPT="${OPENCLAW_BT_SETUP_SCRIPT:-$SETUP_SCRIPT_DEFAULT}"

AUTO_FIX="false"
case "${AUTO_FIX_RAW,,}" in
  1|true|yes|y) AUTO_FIX="true" ;;
esac

if ! id "$BT_USER" >/dev/null 2>&1; then
  echo "FAIL: user '$BT_USER' not found"
  exit 1
fi

BT_USER_UID="$(id -u "$BT_USER")"
USER_SYSTEMCTL=(
  sudo -u "$BT_USER"
  XDG_RUNTIME_DIR="/run/user/$BT_USER_UID"
  DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$BT_USER_UID/bus"
  systemctl --user
)

CHECK_KEYS=(
  "bluetooth.service"
  "bluetooth-host-mode.service"
  "bluez-auto-agent.service"
  "adapter.alias"
  "adapter.discoverable"
  "adapter.pairable"
  "route.hook.executable"
  "openclaw-bt-route.timer.enabled"
  "openclaw-bt-route.timer.active"
)

declare -A CHECK_LABEL
CHECK_LABEL["bluetooth.service"]="bluetooth.service active"
CHECK_LABEL["bluetooth-host-mode.service"]="bluetooth-host-mode.service active"
CHECK_LABEL["bluez-auto-agent.service"]="bluez-auto-agent.service active"
CHECK_LABEL["adapter.alias"]="bluetooth alias"
CHECK_LABEL["adapter.discoverable"]="bluetooth discoverable"
CHECK_LABEL["adapter.pairable"]="bluetooth pairable"
CHECK_LABEL["route.hook.executable"]="route hook executable"
CHECK_LABEL["openclaw-bt-route.timer.enabled"]="openclaw-bt-route.timer enabled"
CHECK_LABEL["openclaw-bt-route.timer.active"]="openclaw-bt-route.timer active"

declare -A FIRST_PASS SECOND_PASS DETAIL
declare -A SUMMARY_STATUS

all_checks_pass() {
  local pass="true"
  local key
  for key in "${CHECK_KEYS[@]}"; do
    if [[ "${SECOND_PASS[$key]:-false}" != "true" ]]; then
      pass="false"
      break
    fi
  done
  [[ "$pass" == "true" ]]
}

record_check() {
  local phase="$1"
  local key="$2"
  local ok="$3"
  local detail="$4"

  DETAIL["$key"]="$detail"

  if [[ "$phase" == "first" ]]; then
    FIRST_PASS["$key"]="$ok"
  else
    SECOND_PASS["$key"]="$ok"
  fi

  if [[ "$ok" == "true" ]]; then
    printf 'PASS: %s%s\n' "${CHECK_LABEL[$key]}" "${detail:+ ($detail)}"
  else
    printf 'FAIL: %s%s\n' "${CHECK_LABEL[$key]}" "${detail:+ ($detail)}"
  fi
}

check_eq() {
  local phase="$1"
  local key="$2"
  local actual="$3"
  local expected="$4"
  local detail="expected=${expected} actual=${actual}"
  if [[ "$actual" == "$expected" ]]; then
    record_check "$phase" "$key" "true" "$detail"
  else
    record_check "$phase" "$key" "false" "$detail"
  fi
}

check_nonempty() {
  local phase="$1"
  local key="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    record_check "$phase" "$key" "true" "value=${value}"
  else
    record_check "$phase" "$key" "false" "value is empty"
  fi
}

run_checks() {
  local phase="$1"

  bt_active="$(systemctl is-active bluetooth 2>/dev/null || true)"
  host_mode_active="$(systemctl is-active bluetooth-host-mode.service 2>/dev/null || true)"
  agent_active="$(systemctl is-active bluez-auto-agent.service 2>/dev/null || true)"

  check_eq "$phase" "bluetooth.service" "$bt_active" "active"
  check_eq "$phase" "bluetooth-host-mode.service" "$host_mode_active" "active"
  check_eq "$phase" "bluez-auto-agent.service" "$agent_active" "active"

  alias_value="$(bluetoothctl show 2>/dev/null | sed -n 's/^[[:space:]]*Alias: //p' | head -n1)"
  discoverable_value="$(bluetoothctl show 2>/dev/null | sed -n 's/^[[:space:]]*Discoverable: //p' | head -n1)"
  pairable_value="$(bluetoothctl show 2>/dev/null | sed -n 's/^[[:space:]]*Pairable: //p' | head -n1)"

  check_nonempty "$phase" "adapter.alias" "$alias_value"
  if [[ -n "$EXPECTED_ALIAS" ]]; then
    if [[ "$alias_value" != "$EXPECTED_ALIAS" ]]; then
      SECOND_PASS["adapter.alias"]="false"
      if [[ "$phase" == "first" ]]; then
        FIRST_PASS["adapter.alias"]="false"
      fi
      DETAIL["adapter.alias"]="expected=${EXPECTED_ALIAS} actual=${alias_value}"
      printf 'FAIL: bluetooth alias target (expected=%s actual=%s)\n' "$EXPECTED_ALIAS" "$alias_value"
    fi
  fi
  check_eq "$phase" "adapter.discoverable" "$discoverable_value" "yes"
  check_eq "$phase" "adapter.pairable" "$pairable_value" "yes"

  route_script="/usr/local/bin/openclaw-bt-route-sink.sh"
  if [[ -x "$route_script" ]]; then
    record_check "$phase" "route.hook.executable" "true" "$route_script"
  else
    record_check "$phase" "route.hook.executable" "false" "$route_script"
  fi

  timer_enabled="$(${USER_SYSTEMCTL[@]} is-enabled openclaw-bt-route.timer 2>/dev/null || true)"
  timer_active="$(${USER_SYSTEMCTL[@]} is-active openclaw-bt-route.timer 2>/dev/null || true)"
  check_eq "$phase" "openclaw-bt-route.timer.enabled" "$timer_enabled" "enabled"
  check_eq "$phase" "openclaw-bt-route.timer.active" "$timer_active" "active"

  if [[ "$phase" == "first" ]]; then
    local key
    for key in "${CHECK_KEYS[@]}"; do
      SECOND_PASS["$key"]="${FIRST_PASS[$key]:-false}"
    done
  fi
}

emit_summary() {
  local auto_fixed="$1"
  local overall="PASS"
  local summary_parts=()
  local key first second status

  for key in "${CHECK_KEYS[@]}"; do
    first="${FIRST_PASS[$key]:-false}"
    second="${SECOND_PASS[$key]:-false}"

    if [[ "$second" == "true" ]]; then
      if [[ "$auto_fixed" == "true" && "$first" != "true" ]]; then
        status="REPAIRED"
      else
        status="PASS"
      fi
    else
      status="FAIL"
      overall="FAIL"
    fi

    SUMMARY_STATUS["$key"]="$status"
    summary_parts+=("${key}=${status}")
    echo "CHECK_STATUS ${key} ${status}"
  done

  echo "BT_SMOKE_SUMMARY ${summary_parts[*]}"
  echo "RESULT: ${overall}"

  if [[ "$overall" == "PASS" ]]; then
    return 0
  fi
  return 1
}

attempt_fix() {
  echo "INFO: attempting auto-fix for missing Bluetooth audio host pieces..."

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "WARN: auto-fix requested but script is not running as root"
    return 1
  fi

  systemctl enable --now bluetooth.service >/dev/null 2>&1 || true
  systemctl enable --now bluetooth-host-mode.service >/dev/null 2>&1 || true
  systemctl enable --now bluez-auto-agent.service >/dev/null 2>&1 || true

  if [[ -x /usr/local/bin/bluetooth-host-mode.sh ]]; then
    /usr/local/bin/bluetooth-host-mode.sh >/dev/null 2>&1 || true
  fi

  if [[ -e /usr/local/bin/openclaw-bt-route-sink.sh ]]; then
    chmod +x /usr/local/bin/openclaw-bt-route-sink.sh || true
  fi

  loginctl enable-linger "$BT_USER" >/dev/null 2>&1 || true
  ${USER_SYSTEMCTL[@]} daemon-reload >/dev/null 2>&1 || true
  ${USER_SYSTEMCTL[@]} enable --now openclaw-bt-route.timer >/dev/null 2>&1 || true
  ${USER_SYSTEMCTL[@]} start openclaw-bt-route.service >/dev/null 2>&1 || true

  if [[ -f "$SETUP_SCRIPT" ]]; then
    OPENCLAW_BT_USER="$BT_USER" OPENCLAW_BT_ALIAS="$EXPECTED_ALIAS" bash "$SETUP_SCRIPT" >/dev/null 2>&1 || true
  fi
}

echo "INFO: running Bluetooth audio host checks"
run_checks "first"
if emit_summary "false"; then
  exit 0
fi

if [[ "$AUTO_FIX" == "true" ]]; then
  attempt_fix
  sleep 2
  echo "INFO: re-running checks after auto-fix"
  run_checks "second"
  if emit_summary "true"; then
    exit 0
  fi
fi

exit 1