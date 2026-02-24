import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import sounddevice as sd
import numpy as np

from orchestrator.config import VoiceConfig
from orchestrator.audio.resample import resample_pcm
from orchestrator.wakeword.openwakeword import OpenWakeWordDetector


class WakeWordLiveHandler(BaseHTTPRequestHandler):
    detector = None
    config = None

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/test/wakeword-live":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        duration_sec = 2.0
        try:
            length_header = self.headers.get("X-Duration-Seconds")
            if length_header:
                duration_sec = float(length_header)
        except ValueError:
            duration_sec = 2.0

        config = self.config
        sample_rate = config.audio_sample_rate
        frame_samples = int(sample_rate * duration_sec)

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

        result = self.detector.detect(pcm)
        self._send_json({"ok": True, "detected": result.detected, "confidence": result.confidence})


def build_detector(config: VoiceConfig):
    return OpenWakeWordDetector(
        model_path=config.openwakeword_model_path,
        confidence=config.wake_word_confidence,
    )


def main() -> None:
    config = VoiceConfig()
    WakeWordLiveHandler.detector = build_detector(config)
    WakeWordLiveHandler.config = config

    server = HTTPServer(("0.0.0.0", 18952), WakeWordLiveHandler)
    print("Wake word live test server listening on :18952 (POST /test/wakeword-live)")
    server.serve_forever()


if __name__ == "__main__":
    main()
