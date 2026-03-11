"""Embedded HTTP + WebSocket service for realtime voice UI telemetry."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import websockets

logger = logging.getLogger("orchestrator.web.realtime")


def _build_ui_html(ws_port: int) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>OpenClaw Voice Realtime</title>
  <style>
    :root {{
      --bg: #0f1117;
      --panel: #171b26;
      --text: #e8ecff;
      --muted: #9aa3c7;
      --ok: #2ad38b;
      --warn: #ffbf47;
      --bad: #ff5d73;
      --bar: #3e4d8a;
      --bar-live: #49c8ff;
      --bar-browser: #7bff9d;
      --chip-off: #2f3548;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; font-family: Inter, system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 860px; margin: 0 auto; padding: 16px; }}
    .card {{ background: var(--panel); border: 1px solid #242a3b; border-radius: 14px; padding: 14px; margin-bottom: 12px; }}
    h1 {{ margin: 0 0 6px; font-size: 1.05rem; }}
    .muted {{ color: var(--muted); font-size: 0.9rem; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    button {{ border: 0; border-radius: 10px; padding: 8px 12px; font-weight: 600; cursor: pointer; background: #2b3f99; color: #fff; }}
    button.stop {{ background: #8f2c40; }}
    .chip {{ padding: 6px 10px; border-radius: 999px; background: var(--chip-off); color: var(--muted); font-size: 0.82rem; }}
    .chip.on {{ color: #001f12; background: var(--ok); }}
    .chip.warn {{ color: #332400; background: var(--warn); }}
    .chip.bad {{ color: #33000a; background: var(--bad); }}
    .meter {{ width: 100%; height: 14px; background: #20263a; border-radius: 999px; overflow: hidden; }}
    .fill {{ height: 100%; width: 0%; background: var(--bar); transition: width 55ms linear; }}
    .fill.orch {{ background: var(--bar-live); }}
    .fill.browser {{ background: var(--bar-browser); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 8px; }}
    label {{ color: var(--muted); font-size: .9rem; }}
    code {{ color: #b9c5ff; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\" id=\"hero\">
      <h1>🎙️ OpenClaw Continuous Voice</h1>
      <div class=\"muted\">Embeddable browser audio streaming + realtime orchestrator status.</div>
      <div class=\"row\" style=\"margin-top:10px\">
        <button id=\"startBtn\">Start continuous audio</button>
        <button id=\"stopBtn\" class=\"stop\" disabled>Stop</button>
        <label><input type=\"checkbox\" id=\"rawPcm\" checked /> stream raw PCM</label>
      </div>
      <div class=\"row\" style=\"margin-top:10px\">
        <span id=\"wsChip\" class=\"chip bad\">ws: disconnected</span>
        <span id=\"capChip\" class=\"chip\">capture: stopped</span>
        <span id=\"stateChip\" class=\"chip\">state: unknown</span>
      </div>
    </div>

    <div class=\"card\">
      <div class=\"muted\" style=\"margin-bottom:6px\">Browser microphone VU</div>
      <div class=\"meter\"><div id=\"browserFill\" class=\"fill browser\"></div></div>
      <div class=\"muted\" id=\"browserText\" style=\"margin-top:6px\">rms: 0.0000</div>
    </div>

    <div class=\"card\">
      <div class=\"muted\" style=\"margin-bottom:6px\">Orchestrator microphone VU</div>
      <div class=\"meter\"><div id=\"orchFill\" class=\"fill orch\"></div></div>
      <div class=\"muted\" id=\"orchText\" style=\"margin-top:6px\">rms: 0.0000</div>
      <div class=\"grid\">
        <span id=\"sleepChip\" class=\"chip\">sleep: ?</span>
        <span id=\"speechChip\" class=\"chip\">speech: ?</span>
        <span id=\"hotwordChip\" class=\"chip\">hotword: ?</span>
        <span id=\"ttsChip\" class=\"chip\">tts: ?</span>
      </div>
    </div>

    <div class=\"card\">
      <div class=\"muted\">Embed this page in your app:</div>
      <code>&lt;iframe src=\"http://HOST:{ws_port - 1}/\" style=\"width:100%;height:420px;border:0\"&gt;&lt;/iframe&gt;</code>
    </div>
  </div>

  <script>
    const WS_PORT = {ws_port};
    const wsChip = document.getElementById('wsChip');
    const capChip = document.getElementById('capChip');
    const stateChip = document.getElementById('stateChip');
    const sleepChip = document.getElementById('sleepChip');
    const speechChip = document.getElementById('speechChip');
    const hotwordChip = document.getElementById('hotwordChip');
    const ttsChip = document.getElementById('ttsChip');
    const browserFill = document.getElementById('browserFill');
    const orchFill = document.getElementById('orchFill');
    const browserText = document.getElementById('browserText');
    const orchText = document.getElementById('orchText');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const rawPcm = document.getElementById('rawPcm');

    let ws = null;
    let audioCtx = null;
    let mediaStream = null;
    let source = null;
    let processor = null;
    let zeroGain = null;
    let lastLevelSent = 0;

    function meterPercent(rms) {{
      const clamped = Math.max(0, Math.min(1, rms || 0));
      return Math.round(Math.min(100, Math.pow(clamped, 0.6) * 120));
    }}

    function setChip(el, on, text, warn=false) {{
      el.textContent = text;
      el.classList.remove('on', 'warn', 'bad');
      if (warn) el.classList.add('warn');
      else if (on) el.classList.add('on');
      else el.classList.add('bad');
    }}

    function wsUrl() {{
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      return `${{proto}}://${{location.hostname}}:${{WS_PORT}}/ws`;
    }}

    function connectWs() {{
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
      ws = new WebSocket(wsUrl());
      ws.binaryType = 'arraybuffer';
      setChip(wsChip, false, 'ws: connecting', true);

      ws.onopen = () => setChip(wsChip, true, 'ws: connected');
      ws.onclose = () => {{
        setChip(wsChip, false, 'ws: disconnected');
        setTimeout(connectWs, 1500);
      }};
      ws.onerror = () => setChip(wsChip, false, 'ws: error');
      ws.onmessage = (evt) => {{
        try {{
          const msg = JSON.parse(evt.data);
          if (msg.type !== 'status') return;
          const orch = msg.orchestrator || {{}};

          const rms = Number(orch.mic_rms || 0);
          orchFill.style.width = `${{meterPercent(rms)}}%`;
          orchText.textContent = `rms: ${{rms.toFixed(4)}}`;

          const wake = String(orch.wake_state || 'unknown');
          const voiceState = String(orch.voice_state || 'unknown');
          const speech = !!orch.speech_active;
          const hotword = !!orch.hotword_active;
          const tts = !!orch.tts_playing;

          stateChip.textContent = `state: ${{voiceState}}`;
          stateChip.className = 'chip on';

          setChip(sleepChip, wake !== 'asleep', `sleep: ${{wake === 'asleep' ? 'asleep' : 'awake'}}`);
          setChip(speechChip, speech, `speech: ${{speech ? 'active' : 'idle'}}`);
          setChip(hotwordChip, hotword, `hotword: ${{hotword ? 'detected' : 'idle'}}`);
          setChip(ttsChip, tts, `tts: ${{tts ? 'speaking' : 'idle'}}`);
        }} catch (_) {{}}
      }};
    }}

    function floatToInt16Buffer(float32) {{
      const out = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {{
        const s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }}
      return out.buffer;
    }}

    async function startCapture() {{
      connectWs();
      mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true, video: false }});
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(2048, 1, 1);
      zeroGain = audioCtx.createGain();
      zeroGain.gain.value = 0;

      source.connect(processor);
      processor.connect(zeroGain);
      zeroGain.connect(audioCtx.destination);

      processor.onaudioprocess = (event) => {{
        const input = event.inputBuffer.getChannelData(0);
        let sumSq = 0;
        let peak = 0;
        for (let i = 0; i < input.length; i++) {{
          const v = input[i];
          sumSq += v * v;
          const av = Math.abs(v);
          if (av > peak) peak = av;
        }}
        const rms = Math.sqrt(sumSq / Math.max(1, input.length));
        browserFill.style.width = `${{meterPercent(rms)}}%`;
        browserText.textContent = `rms: ${{rms.toFixed(4)}}`;

        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const now = performance.now();
        if (now - lastLevelSent >= 120) {{
          lastLevelSent = now;
          ws.send(JSON.stringify({{ type: 'browser_audio_level', rms, peak }}));
        }}

        if (rawPcm.checked) {{
          ws.send(floatToInt16Buffer(input));
        }}
      }};

      setChip(capChip, true, 'capture: running');
      startBtn.disabled = true;
      stopBtn.disabled = false;
    }}

    async function stopCapture() {{
      if (processor) {{ try {{ processor.disconnect(); }} catch (_) {{}} processor = null; }}
      if (source) {{ try {{ source.disconnect(); }} catch (_) {{}} source = null; }}
      if (zeroGain) {{ try {{ zeroGain.disconnect(); }} catch (_) {{}} zeroGain = null; }}
      if (audioCtx) {{ try {{ await audioCtx.close(); }} catch (_) {{}} audioCtx = null; }}
      if (mediaStream) {{
        for (const track of mediaStream.getTracks()) track.stop();
        mediaStream = null;
      }}
      setChip(capChip, false, 'capture: stopped');
      startBtn.disabled = false;
      stopBtn.disabled = true;
      browserFill.style.width = '0%';
      browserText.textContent = 'rms: 0.0000';
    }}

    startBtn.addEventListener('click', () => startCapture().catch((err) => {{
      setChip(capChip, false, 'capture: permission/error');
      console.error(err);
    }}));
    stopBtn.addEventListener('click', () => stopCapture());

    if (window.top !== window.self) {{
      document.getElementById('hero').querySelector('h1').textContent = 'OpenClaw Voice Widget';
    }}

    connectWs();
  </script>
</body>
</html>
"""


class EmbeddedVoiceWebService:
    """Small embedded HTTP/WebSocket service for realtime UI and audio streaming."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        ui_port: int = 18910,
        ws_port: int = 18911,
        status_hz: int = 12,
        hotword_active_ms: int = 2000,
    ):
        self.host = host
        self.ui_port = ui_port
        self.ws_port = ws_port
        self.status_interval_s = 1.0 / max(1, status_hz)
        self.hotword_active_s = max(0.1, hotword_active_ms / 1000.0)

        self._http_server: HTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_server: Any = None
        self._status_task: asyncio.Task | None = None

        self._clients: set[Any] = set()
        self._latest_browser_audio: dict[str, float] = {"rms": 0.0, "peak": 0.0}
        self._last_hotword_ts: float | None = None
        self._orchestrator_status: dict[str, Any] = {
            "voice_state": "idle",
            "wake_state": "asleep",
            "speech_active": False,
            "tts_playing": False,
            "mic_rms": 0.0,
            "queue_depth": 0,
        }

    async def start(self) -> None:
        self._start_http_server()
        self._ws_server = await websockets.serve(self._ws_handler, self.host, self.ws_port)
        self._status_task = asyncio.create_task(self._status_loop())
        logger.info(
            "Embedded web UI started: http://%s:%d (ws://%s:%d)",
            self.host,
            self.ui_port,
            self.host,
            self.ws_port,
        )

    async def stop(self) -> None:
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            self._status_task = None

        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None

        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=1.0)
        self._http_thread = None

        self._clients.clear()

    def update_orchestrator_status(self, **status: Any) -> None:
        self._orchestrator_status.update(status)

    def note_hotword_detected(self) -> None:
        self._last_hotword_ts = time.monotonic()

    async def _ws_handler(self, websocket: Any) -> None:
        client_id = uuid.uuid4().hex[:8]
        self._clients.add(websocket)
        logger.info("Web UI client connected (%s); clients=%d", client_id, len(self._clients))
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "hello",
                        "client_id": client_id,
                        "ws_port": self.ws_port,
                        "ui_port": self.ui_port,
                    }
                )
            )
            async for message in websocket:
                if isinstance(message, str):
                    self._handle_text_message(message)
                elif isinstance(message, (bytes, bytearray)):
                    self._handle_pcm_chunk(bytes(message))
        except Exception as exc:
            logger.debug("Web UI client %s disconnected: %s", client_id, exc)
        finally:
            self._clients.discard(websocket)
            logger.info("Web UI client disconnected (%s); clients=%d", client_id, len(self._clients))

    def _handle_text_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        if payload.get("type") == "browser_audio_level":
            try:
                self._latest_browser_audio["rms"] = float(payload.get("rms", 0.0))
                self._latest_browser_audio["peak"] = float(payload.get("peak", 0.0))
            except Exception:
                pass

    def _handle_pcm_chunk(self, pcm_bytes: bytes) -> None:
        if len(pcm_bytes) < 2:
            return

        sample_count = len(pcm_bytes) // 2
        if sample_count <= 0:
            return

        pcm_view = memoryview(pcm_bytes)[: sample_count * 2].cast("h")

        sum_sq = 0.0
        peak = 0
        for sample in pcm_view:
            s = int(sample)
            abs_sample = -s if s < 0 else s
            if abs_sample > peak:
                peak = abs_sample
            sum_sq += float(s * s)

        rms = math.sqrt(sum_sq / float(sample_count)) / 32768.0
        peak_norm = float(peak) / 32768.0

        self._latest_browser_audio["rms"] = max(0.0, min(1.0, rms))
        self._latest_browser_audio["peak"] = max(0.0, min(1.0, peak_norm))

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self.status_interval_s)
            if not self._clients:
                continue

            payload = self._build_status_payload()
            message = json.dumps(payload)

            stale_clients: list[Any] = []
            for client in list(self._clients):
                try:
                    await client.send(message)
                except Exception:
                    stale_clients.append(client)

            for client in stale_clients:
                self._clients.discard(client)

    def _build_status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        hotword_active = (
            self._last_hotword_ts is not None
            and (now - self._last_hotword_ts) <= self.hotword_active_s
        )

        orchestrator = dict(self._orchestrator_status)
        orchestrator["hotword_active"] = hotword_active

        return {
            "type": "status",
            "ts": time.time(),
            "orchestrator": orchestrator,
            "browser_audio": dict(self._latest_browser_audio),
            "connections": {"websocket_clients": len(self._clients)},
        }

    def _start_http_server(self) -> None:
        html = _build_ui_html(self.ws_port)

        class UIHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def _send(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._send(b"", status=204, content_type="text/plain")

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    self._send(html.encode("utf-8"))
                    return
                if self.path == "/health":
                    payload = json.dumps({"status": "ok", "service": "embedded-voice-ui"}).encode("utf-8")
                    self._send(payload, content_type="application/json")
                    return
                self._send(b"Not found", status=404, content_type="text/plain")

        self._http_server = HTTPServer((self.host, self.ui_port), UIHandler)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
