import logging
from typing import Optional

import numpy as np

from orchestrator.metrics import WakeWordResult
from orchestrator.wakeword.base import WakeWordBase

try:
    from openwakeword.model import Model
except ImportError:  # pragma: no cover
    Model = None


logger = logging.getLogger("orchestrator.wakeword.openwakeword")


class OpenWakeWordDetector(WakeWordBase):
    def __init__(self, model_path: str, confidence: float = 0.5) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Optional[Model] = None
        self._warned = False

        if not Model:
            return

        try:
            # If a specific model path is provided, use it; otherwise load defaults
            if model_path:
                self._model = Model(wakeword_models=[model_path])
            else:
                # Load default pre-trained models (alexa, hey_mycroft, hey_jarvis, timer, weather)
                self._model = Model()
                logger.info("OpenWakeWord loaded default models: %s", list(self._model.models.keys()))
        except Exception as exc:  # pragma: no cover
            logger.warning("OpenWakeWord initialization failed: %s", exc)
            self._model = None

    def detect(self, pcm_frame: bytes) -> WakeWordResult:
        if not self._model:
            if not self._warned:
                logger.warning("OpenWakeWord model not loaded; wake word disabled.")
                self._warned = True
            return WakeWordResult(detected=False, confidence=0.0)

        audio = np.frombuffer(pcm_frame, dtype=np.int16)
        try:
            scores = self._model.predict(audio)
        except Exception as exc:  # pragma: no cover
            logger.warning("OpenWakeWord inference failed: %s", exc)
            return WakeWordResult(detected=False, confidence=0.0)

        if not scores:
            return WakeWordResult(detected=False, confidence=0.0)

        max_score = max(scores.values()) if isinstance(scores, dict) else float(scores)
        detected = max_score >= self.confidence
        return WakeWordResult(detected=detected, confidence=float(max_score))
