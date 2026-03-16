from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ClientState:
    connected: bool = False
    mic_enabled: bool = False
    wake_state: str = "asleep"
    mic_rms: float = 0.0
    tts_muted: bool = False
    continuous_mode: bool = False
    browser_audio_enabled: bool = True
    last_error: str = ""
