from typing import Callable, Optional
import threading

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

    def play_pcm(self, pcm: bytes, gain: float = 1.0, stop_event: Optional[threading.Event] = None) -> None:
        if gain != 1.0:
            pcm = apply_gain(pcm, gain)
        data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        data = data.reshape(-1, 1)
        if self._on_playback_frame:
            self._on_playback_frame(pcm)
        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=None if self.device == "default" else self.device,
            )
            self._stream.start()
        chunk_size = 1024
        total = data.shape[0]
        idx = 0
        while idx < total:
            if stop_event is not None and stop_event.is_set():
                break
            end = min(idx + chunk_size, total)
            self._stream.write(data[idx:end])
            idx = end
