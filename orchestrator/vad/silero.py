from typing import Optional
import logging

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from orchestrator.metrics import VADResult
from orchestrator.vad.base import VADBase


logger = logging.getLogger("orchestrator.vad.silero")


class SileroVAD(VADBase):
    def __init__(self, sample_rate: int, frame_samples: int, model_path: Optional[str] = None) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.model_path = model_path
        self._session = None
        self._warned_sample_rate = False
        self._warned_model = False
        self._input_name = None
        self._sr_input_name = None

        if ort and model_path:
            self._session = ort.InferenceSession(model_path)
            inputs = self._session.get_inputs()
            if inputs:
                self._input_name = inputs[0].name
                for inp in inputs[1:]:
                    if "sr" in inp.name or "sample" in inp.name:
                        self._sr_input_name = inp.name

    def is_speech(self, pcm_frame: bytes) -> VADResult:
        if not self._session or not self._input_name:
            if not self._warned_model:
                logger.warning("Silero VAD model not loaded; returning no speech.")
                self._warned_model = True
            return VADResult(speech_detected=False, confidence=0.0)

        if self.sample_rate != 16000:
            if not self._warned_sample_rate:
                logger.warning("Silero VAD expects 16kHz audio; got %s Hz.", self.sample_rate)
                self._warned_sample_rate = True
            return VADResult(speech_detected=False, confidence=0.0)

        audio = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0

        allowed_sizes = (512, 1024, 1536)
        if audio.size not in allowed_sizes:
            target = next((s for s in allowed_sizes if s >= audio.size), allowed_sizes[-1])
            if audio.size < target:
                audio = np.pad(audio, (0, target - audio.size))
            else:
                audio = audio[:target]

        audio = audio.reshape(1, -1)

        inputs = {self._input_name: audio}
        if self._sr_input_name:
            inputs[self._sr_input_name] = np.array([self.sample_rate], dtype=np.int64)

        outputs = self._session.run(None, inputs)
        confidence = float(outputs[0][0][0]) if outputs else 0.0
        return VADResult(speech_detected=confidence >= 0.5, confidence=confidence)
