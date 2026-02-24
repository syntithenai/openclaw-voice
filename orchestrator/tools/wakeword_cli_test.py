import argparse
import time

import numpy as np
import sounddevice as sd

from orchestrator.audio.resample import resample_pcm
from orchestrator.config import VoiceConfig
from orchestrator.wakeword.openwakeword import OpenWakeWordDetector


def build_detector(config: VoiceConfig):
    return OpenWakeWordDetector(
        model_path=config.openwakeword_model_path,
        confidence=config.wake_word_confidence,
    )


def main() -> None:
    config = VoiceConfig()
    parser = argparse.ArgumentParser(description="Live wakeword capture test")
    parser.add_argument("--duration", type=float, default=2.0, help="Capture duration in seconds")
    args = parser.parse_args()

    detector = build_detector(config)
    sample_rate = config.audio_sample_rate
    frame_samples = int(sample_rate * args.duration)

    print(f"Recording {args.duration}s for wakeword test...")
    recording = sd.rec(
        frames=frame_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=None if config.audio_capture_device == "default" else config.audio_capture_device,
    )
    sd.wait()

    pcm = recording.reshape(-1).tobytes()
    if sample_rate != 16000:
        pcm = resample_pcm(pcm, sample_rate, 16000)

    result = detector.detect(pcm)
    status = "DETECTED" if result.detected else "not detected"
    print(f"Wakeword {status} (confidence={result.confidence:.3f})")
    time.sleep(0.05)


if __name__ == "__main__":
    main()
