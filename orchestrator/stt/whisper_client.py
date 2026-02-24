import requests


class WhisperClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def transcribe(self, wav_bytes: bytes) -> str:
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        response = requests.post(f"{self.base_url}/transcribe", files=files, timeout=120)
        response.raise_for_status()
        payload = response.json()
        return payload.get("text", "")
