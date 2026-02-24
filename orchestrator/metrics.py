from dataclasses import dataclass


@dataclass
class VADResult:
    speech_detected: bool
    confidence: float = 0.0


@dataclass
class WakeWordResult:
    detected: bool
    confidence: float = 0.0


@dataclass
class AECStatus:
    enabled: bool
    backend: str
    available: bool
