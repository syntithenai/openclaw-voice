from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

import websockets


class RealtimeBridge:
    def __init__(
        self,
        ws_url: str,
        reconnect_delay_s: float = 1.5,
        on_message: Callable[[dict[str, Any]], None] | None = None,
        on_connection_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.ws_url = ws_url
        self.reconnect_delay_s = reconnect_delay_s
        self.on_message = on_message
        self.on_connection_change = on_connection_change

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt = threading.Event()
        self._ws: Any | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._loop:
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._close_ws()))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def send_action(self, payload: dict[str, Any]) -> None:
        if not self._loop:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._send(payload)))

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    if self.on_connection_change:
                        self.on_connection_change(True)
                    await ws.send(json.dumps({"type": "ui_ready"}))
                    async for message in ws:
                        if self._stop_evt.is_set():
                            break
                        if isinstance(message, str):
                            try:
                                payload = json.loads(message)
                            except json.JSONDecodeError:
                                continue
                            if self.on_message:
                                self.on_message(payload)
            except Exception:
                pass
            finally:
                self._ws = None
                if self.on_connection_change:
                    self.on_connection_change(False)
            await asyncio.sleep(max(0.2, float(self.reconnect_delay_s)))

    async def _send(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(payload))
        except Exception:
            return

    async def _close_ws(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            return
