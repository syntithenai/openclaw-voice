import json
import io
import wave
import base64
from email import message_from_binary_file
from typing import Tuple

import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

from orchestrator.audio.webrtc_aec import WebRTCAEC
from orchestrator.audio.resample import resample_pcm
from orchestrator.config import VoiceConfig


class AECTestHandler(BaseHTTPRequestHandler):
    aec = None
    config = None

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/test/aec":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json({"ok": False, "error": "No body"}, status=400)
            return

        mic_bytes, playback_bytes, error = self._parse_request(content_length)
        if error:
            self._send_json({"ok": False, "error": error}, status=400)
            return

        try:
            mic_pcm, mic_sr = self._decode_wav(mic_bytes)
            pb_pcm, pb_sr = self._decode_wav(playback_bytes)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Decode failed: {exc}"}, status=400)
            return

        target_sr = self.config.audio_sample_rate
        if mic_sr != target_sr:
            mic_pcm = resample_pcm(mic_pcm, mic_sr, target_sr)
        if pb_sr != target_sr:
            pb_pcm = resample_pcm(pb_pcm, pb_sr, target_sr)

        frame_samples = int(target_sr * (self.config.audio_frame_ms / 1000))
        frame_bytes = frame_samples * 2

        processed_frames = []
        for idx in range(0, min(len(mic_pcm), len(pb_pcm)), frame_bytes):
            mic_frame = mic_pcm[idx:idx + frame_bytes]
            pb_frame = pb_pcm[idx:idx + frame_bytes]
            if len(mic_frame) < frame_bytes or len(pb_frame) < frame_bytes:
                break
            try:
                processed = self.aec.process(mic_frame, pb_frame)
            except NotImplementedError:
                self._send_json({"ok": False, "error": "AEC bindings not available"}, status=500)
                return
            processed_frames.append(processed)

        out_pcm = b"".join(processed_frames)
        out_wav = self._encode_wav(out_pcm, target_sr)
        out_b64 = base64.b64encode(out_wav).decode("utf-8")

        rms_before = self._rms(mic_pcm)
        rms_after = self._rms(out_pcm)
        reduction = ((rms_before - rms_after) / rms_before) if rms_before > 0 else 0.0

        self._send_json({
            "ok": True,
            "processed_wav": out_b64,
            "rms_before": rms_before,
            "rms_after": rms_after,
            "reduction_ratio": reduction,
        })

    def _parse_request(self, content_length: int) -> Tuple[bytes, bytes, str | None]:
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(content_length)

        if content_type.startswith("multipart/form-data"):
            # Parse multipart form data using email module (replacement for deprecated cgi)
            try:
                # Reconstruct HTTP message for email parser
                msg_bytes = (
                    b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
                )
                msg = message_from_binary_file(io.BytesIO(msg_bytes))
                
                mic_bytes = None
                pb_bytes = None
                
                for part in msg.get_payload():
                    if hasattr(part, 'get_filename'):
                        filename = part.get_filename()
                        if filename == "mic" or part.get_param('name') == 'mic':
                            mic_bytes = part.get_payload(decode=True)
                        elif filename == "playback" or part.get_param('name') == 'playback':
                            pb_bytes = part.get_payload(decode=True)
                
                if not mic_bytes or not pb_bytes:
                    return b"", b"", "multipart form requires 'mic' and 'playback' fields"
                
                return mic_bytes, pb_bytes, None
            except Exception as exc:
                return b"", b"", f"Multipart parse failed: {exc}"

        try:
            payload = json.loads(body)
            mic_wav = payload.get("mic_wav")
            playback_wav = payload.get("playback_wav")
        except Exception:
            return b"", b"", "Invalid JSON"

        if not mic_wav or not playback_wav:
            return b"", b"", "mic_wav and playback_wav required (base64)"

        try:
            mic_bytes = base64.b64decode(mic_wav)
            playback_bytes = base64.b64decode(playback_wav)
        except Exception as exc:
            return b"", b"", f"Decode failed: {exc}"

        return mic_bytes, playback_bytes, None
    def _rms(self, pcm: bytes) -> float:
        if not pcm:
            return 0.0
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        samples = samples / 32768.0
        return float(np.sqrt(np.mean(samples * samples)))

    def _decode_wav(self, data: bytes):
        with io.BytesIO(data) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                pcm = wav_file.readframes(wav_file.getnframes())
                sr = wav_file.getframerate()
                return pcm, sr

    def _encode_wav(self, pcm: bytes, sample_rate: int) -> bytes:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm)
            return buffer.getvalue()


def main() -> None:
    config = VoiceConfig()
    aec = WebRTCAEC(
        sample_rate=config.audio_sample_rate,
        frame_ms=config.audio_frame_ms,
        strength=config.echo_cancel_strength,
    )
    AECTestHandler.aec = aec
    AECTestHandler.config = config

    server = HTTPServer(("0.0.0.0", 18951), AECTestHandler)
    print("AEC test server listening on :18951 (POST /test/aec)")
    server.serve_forever()


if __name__ == "__main__":
    main()
