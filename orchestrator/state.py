from enum import Enum


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SENDING = "sending"
    WAITING = "waiting"
    SPEAKING = "speaking"
    ERROR = "error"


class WakeState(str, Enum):
    AWAKE = "awake"
    ASLEEP = "asleep"
