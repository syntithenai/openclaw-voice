import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from orchestrator.audio.resample import resample_pcm
from orchestrator.config import VoiceConfig
from orchestrator.wakeword.openwakeword import OpenWakeWordDetector

import io
import wave


class WakeWordHandler(BaseHTTPRequestHandler):
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
        if self.path != "/test/wakeword":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json({"ok": False, "error": "No body"}, status=400)
            return

        body = self.rfile.read(content_length)
        try:
            with io.BytesIO(body) as buffer:
                with wave.open(buffer, "rb") as wav_file:
                    pcm = wav_file.readframes(wav_file.getnframes())
                    sample_rate = wav_file.getframerate()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid WAV: {exc}"}, status=400)
            return

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
    WakeWordHandler.detector = build_detector(config)
    WakeWordHandler.config = config

    server = HTTPServer(("0.0.0.0", 18950), WakeWordHandler)
    print("Wake word test server listening on :18950 (POST /test/wakeword)")
    server.serve_forever()


if __name__ == "__main__":
    main()
