from abc import ABC, abstractmethod
from orchestrator.metrics import VADResult


class VADBase(ABC):
    @abstractmethod
    def is_speech(self, pcm_frame: bytes) -> VADResult:
        raise NotImplementedError
