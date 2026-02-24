from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from orchestrator.tts.tts_mixer import apply_gain


class AudioPlayback:
    def __init__(self, sample_rate: int, device: str = "default") -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._stream: Optional[sd.OutputStream] = None
        self._on_playback_frame: Optional[Callable[[bytes], None]] = None

    def set_playback_callback(self, callback: Callable[[bytes], None]) -> None:
        self._on_playback_frame = callback

    def play_pcm(self, pcm: bytes, gain: float = 1.0) -> None:
        if gain != 1.0:
            pcm = apply_gain(pcm, gain)
        data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        data = data.reshape(-1, 1)
        if self._on_playback_frame:
            self._on_playback_frame(pcm)
        sd.play(data, samplerate=self.sample_rate, device=None if self.device == "default" else self.device)
        sd.wait()
