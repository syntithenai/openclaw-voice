"""Fake OpenClaw Gateway for testing without actual gateway service."""

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets

logger = logging.getLogger("orchestrator.tools.fake_gateway")


class FakeGatewayHandler(BaseHTTPRequestHandler):
    """HTTP handler for fake gateway responses."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        """Handle POST requests for transcript/response endpoints."""
        if self.path == "/api/short":
            # Short response endpoint
            self._send_json(build_short_response())
        elif self.path == "/api/long":
            # Long response endpoint
            self._send_json(build_long_response())
        else:
            self._send_json({"error": "Not found"}, status=404)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        """Send JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_short_response() -> dict:
    return {
        "ok": True,
        "text": "That's correct.",
        "emotion": "neutral",
        "duration_ms": 500,
    }


def build_long_response() -> dict:
    return {
        "ok": True,
        "text": (
            "I understand you're asking about that. Let me explain in detail. "
            "There are several key points to consider. First, it's important to note "
            "that the system is working as expected. Second, the configuration has been "
            "set up correctly. Third, we've implemented all necessary features. "
            "So in summary, everything is functioning properly."
        ),
        "emotion": "informative",
        "duration_ms": 4500,
    }


async def ws_handler(websocket) -> None:
    async for message in websocket:
        mode = "short"
        try:
            payload = json.loads(message)
            mode = payload.get("endpoint") or payload.get("mode") or payload.get("path") or "short"
            mode = str(mode).replace("/api/", "")
        except json.JSONDecodeError:
            mode = "short"

        response = build_long_response() if mode == "long" else build_short_response()
        await websocket.send(json.dumps(response))


def run_http_server() -> None:
    server = HTTPServer(("0.0.0.0", 18901), FakeGatewayHandler)
    server.serve_forever()


def main() -> None:
    """Start fake gateway server."""
    logger.info("Fake gateway test server listening on :18901 (HTTP)")
    logger.info("Fake gateway test server listening on :18902 (WebSocket)")
    logger.info("  POST /api/short   - Returns short response")
    logger.info("  POST /api/long    - Returns long response")
    logger.info("  GET /health       - Health check")
    logger.info("  WS  ws://localhost:18902  - Send {\"mode\": \"short|long\"}")
    
    try:
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()
        async def ws_main() -> None:
            async with websockets.serve(ws_handler, "0.0.0.0", 18902):
                await asyncio.Future()

        asyncio.run(ws_main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
