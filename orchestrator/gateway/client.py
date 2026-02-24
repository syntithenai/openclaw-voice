import asyncio
import json
import logging
from typing import AsyncIterator

import requests
import websockets


logger = logging.getLogger("orchestrator.gateway")


class GatewayClient:
    def __init__(
        self,
        ws_url: str = "",
        http_url: str = "",
        http_endpoint: str = "/api/short",
        agent_id: str = "",
        auth_token: str = "",
        provider: str = "openclaw",
    ) -> None:
        self.ws_url = ws_url
        self.http_url = http_url.rstrip("/") if http_url else ""
        self.http_endpoint = http_endpoint if http_endpoint.startswith("/") else f"/{http_endpoint}"
        self.agent_id = agent_id
        self.auth_token = auth_token
        self.provider = provider

    async def send_transcript(self, text: str) -> str | None:
        payload = {"type": "transcript", "text": text, "agentId": self.agent_id}
        if self.auth_token:
            payload["token"] = self.auth_token

        if self.ws_url:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(json.dumps(payload))
                return None
            except (ConnectionRefusedError, OSError, websockets.InvalidURI, websockets.InvalidHandshake) as exc:
                logger.warning("Gateway unavailable (%s); attempting HTTP fallback", exc)

        if self.http_url:
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    f"{self.http_url}{self.http_endpoint}",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("text"):
                    return str(data.get("text"))
            except Exception as exc:
                logger.warning("Gateway HTTP unavailable (%s); transcript skipped", exc)
                return None
        return None

    async def listen(self) -> AsyncIterator[str]:
        if not self.ws_url:
            if False:
                yield ""
            return
        try:
            async with websockets.connect(self.ws_url) as ws:
                async for message in ws:
                    yield message
                    await asyncio.sleep(0)
        except (ConnectionRefusedError, OSError, websockets.InvalidURI, websockets.InvalidHandshake) as exc:
            logger.warning("Gateway unavailable (%s); listener disabled", exc)
