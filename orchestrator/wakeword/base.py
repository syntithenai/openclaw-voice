from abc import ABC, abstractmethod
from orchestrator.metrics import WakeWordResult


class WakeWordBase(ABC):
    @abstractmethod
    def detect(self, pcm_frame: bytes) -> WakeWordResult:
        raise NotImplementedError
