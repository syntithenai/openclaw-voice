import asyncio
import json
from typing import AsyncIterator

import websockets


class GatewayClient:
    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url

    async def send_transcript(self, text: str) -> None:
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps({"type": "transcript", "text": text}))

    async def listen(self) -> AsyncIterator[str]:
        async with websockets.connect(self.ws_url) as ws:
            async for message in ws:
                yield message
                await asyncio.sleep(0)
