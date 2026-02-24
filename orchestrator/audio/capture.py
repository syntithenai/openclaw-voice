import queue
from typing import Optional
import logging
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger("orchestrator.audio.capture")


class AudioCapture:
    def __init__(self, sample_rate: int, frame_samples: int, device: str = "default") -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None
        self._warned_status = False

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            if not self._warned_status:
                logger.warning("Audio capture status: %s", status)
                self._warned_status = True
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            pass

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=None if self.device == "default" else self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def restart(self) -> None:
        try:
            self.stop()
            time.sleep(0.05)
            self.start()
            logger.info("Audio capture restarted")
        except Exception as exc:  # pragma: no cover
            logger.warning("Audio capture restart failed: %s", exc)

    def read_frame(self, timeout: float = 0.0) -> Optional[bytes]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
