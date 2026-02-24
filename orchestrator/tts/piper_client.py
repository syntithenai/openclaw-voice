import requests


class PiperClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def synthesize(self, text: str, voice: str = "en_US-amy-medium") -> bytes:
        response = requests.post(
            f"{self.base_url}/synthesize",
            json={"text": text, "voice": voice},
            timeout=120,
        )
        response.raise_for_status()
        return response.content
