import numpy as np


def apply_gain(pcm: bytes, gain: float) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    samples = samples * gain
    samples = np.clip(samples, -32768, 32767)
    return samples.astype(np.int16).tobytes()
