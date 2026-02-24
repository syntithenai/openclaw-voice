import numpy as np


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm
    if src_rate <= 0 or dst_rate <= 0:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return pcm

    duration = samples.size / float(src_rate)
    dst_size = int(duration * dst_rate)
    if dst_size <= 0:
        return pcm

    src_x = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_size, endpoint=False)
    resampled = np.interp(dst_x, src_x, samples)
    resampled = np.clip(resampled, -32768, 32767)
    return resampled.astype(np.int16).tobytes()
