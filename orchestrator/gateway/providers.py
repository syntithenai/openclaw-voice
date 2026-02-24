import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import uuid4
import time

import requests
import websockets


logger = logging.getLogger("orchestrator.gateway.providers")


@dataclass
class GatewayResponse:
    text: str
    run_id: str
    session_id: str


class BaseGateway:
    provider: str = "base"
    supports_listen: bool = False

    async def send_message(
        self,
        text: str,
        session_id: str,
        agent_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        raise NotImplementedError

    async def listen(self) -> AsyncIterator[str]:
        if False:
            yield ""
        return


class GenericGateway(BaseGateway):
    provider = "generic"

    def __init__(
        self,
        http_url: str = "",
        http_endpoint: str = "/api/short",
        ws_url: str = "",
        timeout_s: int = 30,
    ) -> None:
        self.http_url = http_url.rstrip("/") if http_url else ""
        self.http_endpoint = http_endpoint if http_endpoint.startswith("/") else f"/{http_endpoint}"
        self.ws_url = ws_url
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        payload = {
            "text": text,
            "sessionId": session_id,
            "agentId": agent_id,
            "metadata": metadata or {},
        }

        if self.ws_url:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(json.dumps(payload))
                    message = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
                    data = json.loads(message)
                    if isinstance(data, dict):
                        return data.get("text") or data.get("response") or data.get("message")
            except Exception as exc:
                logger.warning("Generic gateway WS failed (%s); attempting HTTP", exc)

        if not self.http_url:
            return None

        def _post() -> requests.Response:
            return requests.post(
                f"{self.http_url}{self.http_endpoint}",
                json=payload,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("text") or data.get("response") or data.get("message")
        return None


class OpenClawGateway(BaseGateway):
    provider = "openclaw"

    def __init__(
        self,
        gateway_url: str,
        token: str,
        agent_id: str = "assistant",
        session_prefix: str = "voice",
        timeout_s: int = 30,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.agent_id = agent_id
        self.session_prefix = session_prefix
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        session_key = f"{self.session_prefix}:{session_id}"
        payload = {
            "sessionKey": session_key,
            "agentId": agent_id or self.agent_id,
            "userMessage": text,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        def _post() -> requests.Response:
            return requests.post(
                f"{self.gateway_url}/hooks/agent",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if isinstance(data.get("message"), str):
                return data.get("message")
            if isinstance(data.get("content"), str):
                return data.get("content")
        return None


class ZeroClawGateway(BaseGateway):
    provider = "zeroclaw"

    def __init__(
        self,
        gateway_url: str,
        webhook_token: str,
        channel: str = "voice",
        timeout_s: int = 30,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.webhook_token = webhook_token
        self.channel = channel
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.webhook_token}",
            "Content-Type": "application/json",
            "X-Session-Id": session_id,
            "X-Agent-Id": agent_id or "default",
            "X-Channel": self.channel,
        }
        payload = {
            "text": text,
            "metadata": {
                "channel": self.channel,
                **(metadata or {}),
            },
        }

        def _post() -> requests.Response:
            return requests.post(
                f"{self.gateway_url}/webhook",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("response") or data.get("text") or data.get("message")
        return None


class TinyClawGateway(BaseGateway):
    provider = "tinyclaw"

    def __init__(
        self,
        tinyclaw_home: str,
        agent_id: str = "default",
        timeout_s: int = 30,
    ) -> None:
        self.tinyclaw_home = Path(tinyclaw_home)
        self.agent_id = agent_id
        self.timeout_s = timeout_s
        self.queue_dir = self.tinyclaw_home / "queue"

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        message_id = f"{int(time.time() * 1000)}-{uuid4()}"
        await asyncio.to_thread(self._ensure_queue_dirs)

        payload = {
            "id": message_id,
            "sessionId": session_id,
            "agentId": agent_id or self.agent_id,
            "message": text,
            "timestamp": int(time.time() * 1000),
            "source": "voice",
            "metadata": metadata or {},
        }

        incoming_path = self.queue_dir / "incoming" / f"{message_id}.json"
        await asyncio.to_thread(incoming_path.write_text, json.dumps(payload, indent=2), "utf-8")

        response = await self._wait_for_response(message_id)
        if isinstance(response, dict):
            return response.get("response") or response.get("message")
        return None

    def _ensure_queue_dirs(self) -> None:
        for name in ("incoming", "outgoing", "processing"):
            (self.queue_dir / name).mkdir(parents=True, exist_ok=True)

    async def _wait_for_response(self, message_id: str) -> dict:
        outgoing_path = self.queue_dir / "outgoing" / f"{message_id}.json"
        start = time.monotonic()
        while time.monotonic() - start < self.timeout_s:
            if outgoing_path.exists():
                data = json.loads(outgoing_path.read_text("utf-8"))
                try:
                    outgoing_path.unlink()
                except OSError:
                    pass
                return data
            await asyncio.sleep(0.1)
        raise TimeoutError(f"TinyClaw response timeout for {message_id}")


class IronClawGateway(BaseGateway):
    provider = "ironclaw"

    def __init__(
        self,
        gateway_url: str,
        token: str,
        agent_id: str = "default",
        use_websocket: bool = True,
        timeout_s: int = 30,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.agent_id = agent_id
        self.use_websocket = use_websocket
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        if self.use_websocket:
            return await self._send_ws(text, session_id, agent_id, metadata)
        return await self._send_http(text, session_id, agent_id, metadata)

    async def _send_ws(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict]) -> Optional[str]:
        ws_url = f"{self.gateway_url.replace('http', 'ws').rstrip('/')}/ws"
        run_id = str(uuid4())
        payload = {
            "type": "message",
            "runId": run_id,
            "sessionId": session_id,
            "agentId": agent_id or self.agent_id,
            "text": text,
            "metadata": metadata or {},
        }
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps(payload))
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
                data = json.loads(message)
                if data.get("runId") == run_id:
                    return data.get("text") or data.get("response")
        return None

    async def _send_http(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict]) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "sessionId": session_id,
            "agentId": agent_id or self.agent_id,
            "text": text,
            "metadata": metadata or {},
        }

        def _post() -> requests.Response:
            return requests.post(
                f"{self.gateway_url}/api/message",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("text") or data.get("response")
        return None


class MimiClawGateway(BaseGateway):
    provider = "mimiclaw"

    def __init__(
        self,
        device_host: str,
        device_port: int = 18789,
        use_websocket: bool = True,
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        timeout_s: int = 30,
    ) -> None:
        self.device_host = device_host
        self.device_port = device_port
        self.use_websocket = use_websocket
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        if self.use_websocket:
            return await self._send_ws(text, session_id, metadata)
        if self.telegram_bot_token and self.telegram_chat_id:
            return await self._send_telegram(text)
        raise RuntimeError("MimiClaw requires WebSocket or Telegram configuration")

    async def _send_ws(self, text: str, session_id: str, metadata: Optional[dict]) -> Optional[str]:
        ws_url = f"ws://{self.device_host}:{self.device_port}"
        request_id = str(uuid4())
        payload = {
            "requestId": request_id,
            "sessionId": session_id,
            "userInput": text,
            "timestamp": int(time.time() * 1000),
            "metadata": metadata or {},
        }
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps(payload))
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
                data = json.loads(message)
                if data.get("requestId") == request_id:
                    return data.get("response") or data.get("message")
        return None

    async def _send_telegram(self, text: str) -> Optional[str]:
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        def _post() -> requests.Response:
            return requests.post(
                f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage",
                json=payload,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        message_id = data.get("result", {}).get("message_id")
        return f"Message sent via Telegram (ID: {message_id})"


class PicoClawGateway(BaseGateway):
    provider = "picoclaw"

    def __init__(
        self,
        workspace_home: str,
        gateway_url: str = "",
        agent_id: str = "default",
        timeout_s: int = 30,
    ) -> None:
        self.workspace_home = Path(workspace_home)
        self.gateway_url = gateway_url.rstrip("/") if gateway_url else ""
        self.agent_id = agent_id
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        if self.gateway_url:
            try:
                return await self._send_http(text, session_id, agent_id, metadata)
            except Exception as exc:
                logger.warning("PicoClaw HTTP gateway failed (%s); falling back to file-based mode", exc)

        sessions_dir = self.workspace_home / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_file = sessions_dir / f"{session_id}.jsonl"
        message_id = f"{int(time.time() * 1000)}-{uuid4()}"
        entry = {
            "id": message_id,
            "timestamp": int(time.time() * 1000),
            "type": "user",
            "content": text,
            "agent": agent_id or self.agent_id,
            "metadata": metadata or {},
        }
        await asyncio.to_thread(session_file.write_text, session_file.read_text("utf-8") + json.dumps(entry) + "\n" if session_file.exists() else json.dumps(entry) + "\n", "utf-8")
        return f"Message queued for {agent_id or self.agent_id} agent (PicoClaw file-based mode)"

    async def _send_http(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict]) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "X-Session-Id": session_id,
            "X-Agent-Id": agent_id or self.agent_id,
        }
        payload = {
            "text": text,
            "metadata": metadata or {},
            "timestamp": int(time.time() * 1000),
        }

        def _post() -> requests.Response:
            return requests.post(
                f"{self.gateway_url}/webhook",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("text") or data.get("response")
        return None


class NanoBotGateway(BaseGateway):
    provider = "nanobot"

    def __init__(
        self,
        workspace_home: str,
        gateway_url: str = "",
        agent_id: str = "",
        timeout_s: int = 30,
    ) -> None:
        self.workspace_home = Path(workspace_home)
        self.gateway_url = gateway_url.rstrip("/") if gateway_url else ""
        self.agent_id = agent_id
        self.timeout_s = timeout_s

    async def send_message(self, text: str, session_id: str, agent_id: str, metadata: Optional[dict] = None) -> Optional[str]:
        if self.gateway_url:
            try:
                return await self._send_http(text, session_id, metadata)
            except Exception as exc:
                logger.warning("NanoBot HTTP gateway failed (%s); falling back to file-based mode", exc)

        workspace_dir = self.workspace_home / "workspace"
        sessions_dir = workspace_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        channel_key = f"voice:{session_id}"
        session_file = sessions_dir / f"{channel_key}.json"
        message_id = f"{int(time.time() * 1000)}-{uuid4()}"

        def _update_session() -> None:
            if session_file.exists():
                session = json.loads(session_file.read_text("utf-8"))
            else:
                session = {
                    "channel": "voice",
                    "chat_id": session_id,
                    "messages": [],
                    "created_at": int(time.time() * 1000),
                    "updated_at": int(time.time() * 1000),
                }
            session["messages"].append({
                "id": message_id,
                "role": "user",
                "content": text,
                "timestamp": int(time.time() * 1000),
                "metadata": metadata or {},
            })
            session["updated_at"] = int(time.time() * 1000)
            session_file.write_text(json.dumps(session, indent=2), "utf-8")

        await asyncio.to_thread(_update_session)
        return "Message queued for NanoBot (file-based mode)"

    async def _send_http(self, text: str, session_id: str, metadata: Optional[dict]) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "X-Session-Id": session_id,
            "X-Channel": "voice",
        }
        payload = {
            "text": text,
            "chat_id": session_id,
            "channel": "voice",
            "metadata": metadata or {},
            "timestamp": int(time.time() * 1000),
        }

        def _post() -> requests.Response:
            return requests.post(
                f"{self.gateway_url}/message",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )

        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get("text") or data.get("response")
        return None
