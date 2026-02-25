import requests


class PiperClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def synthesize(self, text: str, voice: str = "en_US-amy-medium", speed: float = 1.0) -> bytes:
        # Convert speed multiplier to Piper's length_scale (inverse relationship)
        length_scale = 1.0 / speed if speed > 0 else 1.0
        response = requests.post(
            f"{self.base_url}/synthesize",
            json={"text": text, "voice": voice, "length_scale": length_scale},
            timeout=120,
        )
        response.raise_for_status()
        return response.content
