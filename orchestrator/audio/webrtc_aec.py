import logging
from typing import Optional

import numpy as np

try:
    from webrtc_audio_processing import AudioProcessing
except ImportError:  # pragma: no cover
    AudioProcessing = None


logger = logging.getLogger("orchestrator.aec")


class WebRTCAEC:
    def __init__(self, sample_rate: int, frame_ms: int, strength: str = "strong") -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.strength = strength
        self._ap: Optional[AudioProcessing] = None
        self._warned_missing = False

        if not AudioProcessing:
            return

        self._ap = AudioProcessing(
            echo_cancellation=True,
            noise_suppression=True,
            gain_control=True,
            high_pass_filter=True,
        )

        if hasattr(self._ap, "set_stream_format"):
            try:
                self._ap.set_stream_format(
                    self.sample_rate,
                    1,
                    self.sample_rate,
                    1,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("WebRTC AEC set_stream_format failed: %s", exc)

        if hasattr(self._ap, "set_reverse_stream_format"):
            try:
                self._ap.set_reverse_stream_format(self.sample_rate, 1)
            except Exception as exc:  # pragma: no cover
                logger.warning("WebRTC AEC set_reverse_stream_format failed: %s", exc)

    def _frame_samples(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000))

    def _normalize_frame(self, pcm: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm, dtype=np.int16)
        target = self._frame_samples()
        if samples.size < target:
            samples = np.pad(samples, (0, target - samples.size))
        elif samples.size > target:
            samples = samples[:target]
        return samples

    def process(self, mic_pcm: bytes, playback_pcm: bytes) -> bytes:
        if not self._ap:
            if not self._warned_missing:
                logger.warning("WebRTC AEC bindings unavailable. Install webrtc-audio-processing.")
                self._warned_missing = True
            raise NotImplementedError("WebRTC AEC bindings not wired yet")

        mic_samples = self._normalize_frame(mic_pcm)
        pb_samples = self._normalize_frame(playback_pcm)

        try:
            if hasattr(self._ap, "process_reverse_stream"):
                self._ap.process_reverse_stream(pb_samples)
            if hasattr(self._ap, "process_stream"):
                processed = self._ap.process_stream(mic_samples)
                if isinstance(processed, np.ndarray):
                    return processed.astype(np.int16).tobytes()
                if isinstance(processed, (bytes, bytearray)):
                    return bytes(processed)
        except Exception as exc:  # pragma: no cover
            logger.warning("WebRTC AEC processing failed: %s", exc)
            return mic_pcm

        return mic_samples.astype(np.int16).tobytes()
