import webrtcvad

from orchestrator.metrics import VADResult
from orchestrator.vad.base import VADBase


class WebRTCVAD(VADBase):
    def __init__(self, sample_rate: int, frame_ms: int, mode: int = 2) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._vad = webrtcvad.Vad(mode)

    def is_speech(self, pcm_frame: bytes) -> VADResult:
        is_speech = self._vad.is_speech(pcm_frame, self.sample_rate)
        return VADResult(speech_detected=is_speech, confidence=1.0 if is_speech else 0.0)
