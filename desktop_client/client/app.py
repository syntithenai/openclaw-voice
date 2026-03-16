from __future__ import annotations

from .config import load_config
from .controller import TrayController
from .realtime import RealtimeBridge
from .tray import DesktopTrayApp


def run() -> None:
    cfg = load_config()
    bridge = RealtimeBridge(
        ws_url=cfg.ws_url,
        reconnect_delay_s=cfg.reconnect_delay_s,
    )
    controller = TrayController(cfg, bridge)

    bridge.on_message = controller.apply_message
    bridge.on_connection_change = controller.on_connection_change

    app = DesktopTrayApp(controller)

    bridge.start()
    try:
        app.run()
    finally:
        bridge.stop()
