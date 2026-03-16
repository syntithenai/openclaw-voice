from __future__ import annotations

import webbrowser
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Protocol

from .config import DesktopClientConfig
from .state import ClientState


class ActionTransport(Protocol):
    def send_action(self, payload: dict[str, Any]) -> None: ...


class TrayController:
    def __init__(self, config: DesktopClientConfig, transport: ActionTransport) -> None:
        self.config = config
        self.transport = transport
        self.state = ClientState(
            tts_muted=config.default_tts_muted,
            continuous_mode=config.default_continuous_mode,
        )
        self._listeners: list[Callable[[ClientState], None]] = []

    def add_listener(self, listener: Callable[[ClientState], None]) -> None:
        self._listeners.append(listener)

    def _emit(self) -> None:
        for listener in self._listeners:
            listener(self.state)

    def on_connection_change(self, connected: bool) -> None:
        self.state.connected = bool(connected)
        if connected:
            self.push_ui_preferences()
        self._emit()

    def open_web_ui(self) -> None:
        webbrowser.open(self.config.web_ui_url)

    def trigger_mic_toggle(self) -> None:
        self.transport.send_action({"type": "mic_toggle"})

    def toggle_tts_mute(self) -> None:
        self.state.tts_muted = not self.state.tts_muted
        self.transport.send_action({"type": "tts_mute_set", "enabled": self.state.tts_muted})
        self._emit()

    def toggle_continuous_mode(self) -> None:
        self.state.continuous_mode = not self.state.continuous_mode
        self.transport.send_action({"type": "continuous_mode_set", "enabled": self.state.continuous_mode})
        self._emit()

    def push_ui_preferences(self) -> None:
        self.transport.send_action({"type": "tts_mute_set", "enabled": bool(self.state.tts_muted)})
        self.transport.send_action({"type": "continuous_mode_set", "enabled": bool(self.state.continuous_mode)})

    def apply_message(self, msg: dict[str, Any]) -> None:
        msg_type = str(msg.get("type", ""))
        if msg_type == "state_snapshot":
            orch = msg.get("orchestrator") or {}
            ui = msg.get("ui_control") or {}
            self._apply_orchestrator(orch)
            self._apply_ui_control(ui)
            self._emit()
            return
        if msg_type == "orchestrator_status":
            self._apply_orchestrator(msg)
            self._emit()
            return
        if msg_type == "ui_control":
            self._apply_ui_control(msg)
            self._emit()
            return

    def _apply_orchestrator(self, orch: dict[str, Any]) -> None:
        if "mic_rms" in orch:
            try:
                self.state.mic_rms = max(0.0, min(1.0, float(orch.get("mic_rms", 0.0))))
            except Exception:
                pass
        if "wake_state" in orch:
            self.state.wake_state = str(orch.get("wake_state") or self.state.wake_state)
        if "mic_enabled" in orch:
            self.state.mic_enabled = bool(orch.get("mic_enabled"))

    def _apply_ui_control(self, ui: dict[str, Any]) -> None:
        if "tts_muted" in ui:
            self.state.tts_muted = bool(ui.get("tts_muted"))
        if "continuous_mode" in ui:
            self.state.continuous_mode = bool(ui.get("continuous_mode"))
        if "browser_audio_enabled" in ui:
            self.state.browser_audio_enabled = bool(ui.get("browser_audio_enabled"))
        if "mic_enabled" in ui:
            self.state.mic_enabled = bool(ui.get("mic_enabled"))

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.state)
