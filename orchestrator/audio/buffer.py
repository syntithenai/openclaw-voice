from collections import deque
from typing import Deque, List


class RingBuffer:
    def __init__(self, max_frames: int) -> None:
        self._frames: Deque[bytes] = deque(maxlen=max_frames)

    def add_frame(self, frame: bytes) -> None:
        self._frames.append(frame)

    def get_frames(self) -> List[bytes]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()
