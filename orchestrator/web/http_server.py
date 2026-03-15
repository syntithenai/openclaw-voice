from __future__ import annotations

import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def start_http_servers(service: Any, html: str, ssl_context: ssl.SSLContext | None) -> None:
    """Start embedded UI HTTP server and optional HTTP->HTTPS redirector."""

    class UIHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _send(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(b"", status=204, content_type="text/plain")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(html.encode("utf-8"))
            elif path == "/favicon.ico":
                self._send(b"", status=204, content_type="image/x-icon")
            elif path == "/health":
                self._send(
                    json.dumps(
                        {
                            "status": "ok",
                            "service": "embedded-voice-ui",
                            "instance_id": self.server._embedded_instance_id,
                        }
                    ).encode(),
                    content_type="application/json",
                )
            else:
                self._send(b"Not found", status=404, content_type="text/plain")

    class RedirectHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _redirect_target(self) -> str:
            raw_host = self.headers.get("Host", "")
            host = raw_host.split(":", 1)[0].strip() or service.host or "localhost"
            port_suffix = "" if service.ui_port == 443 else f":{service.ui_port}"
            return f"https://{host}{port_suffix}{self.path}"

        def _redirect(self) -> None:
            target = self._redirect_target()
            self.send_response(307)
            self.send_header("Location", target)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            self._redirect()

        def do_HEAD(self) -> None:  # noqa: N802
            self._redirect()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._redirect()

    service._http_server = HTTPServer((service.host, service.ui_port), UIHandler)
    if ssl_context is not None:
        service._http_server.socket = ssl_context.wrap_socket(service._http_server.socket, server_side=True)
    service._http_server._embedded_instance_id = service._instance_id  # type: ignore[attr-defined]
    service._http_thread = threading.Thread(target=service._http_server.serve_forever, daemon=True)
    service._http_thread.start()

    if ssl_context is not None and service.http_redirect_port:
        service._http_redirect_server = HTTPServer((service.host, service.http_redirect_port), RedirectHandler)
        service._http_redirect_thread = threading.Thread(
            target=service._http_redirect_server.serve_forever,
            daemon=True,
        )
        service._http_redirect_thread.start()
