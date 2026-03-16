from __future__ import annotations

import os
import threading
from typing import Any

from .config import derive_ws_url
from .controller import TrayController
from .settings_store import write_desktop_env
from .settings_ui import SettingsDialog
from .state import ClientState
from .vu import border_width_for_state


class DesktopTrayApp:
    def __init__(self, controller: TrayController) -> None:
        self.controller = controller
        self.controller.add_listener(self._on_state_changed)

        self._icon = None
        self._pystray = None
        self._pil_image = None
        self._last_status_line = "OpenClaw Voice Desktop Client"
        self._fallback_opened = False
        self._control_root = None  # tk.Tk reference for deiconify

    def run(self) -> None:
        import pystray
        from PIL import Image, ImageDraw

        self._pystray = pystray
        self._pil_image = (Image, ImageDraw)

        menu = pystray.Menu(
            pystray.MenuItem("Open Controls", self._open_control_window, default=True),
            pystray.MenuItem("Toggle Microphone", self._toggle_mic),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Access Web UI", self._open_web_ui),
            pystray.MenuItem("Mute TTS", self._toggle_tts, checked=lambda _: self.controller.state.tts_muted),
            pystray.MenuItem(
                "Continuous Mode",
                self._toggle_continuous,
                checked=lambda _: self.controller.state.continuous_mode,
            ),
            pystray.MenuItem("Settings", self._open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

        icon = pystray.Icon(
            name="openclaw-desktop-client",
            icon=self._make_icon(self.controller.state),
            title="OpenClaw Voice Desktop Client",
            menu=menu,
        )
        self._icon = icon

        # Run tray detached and keep a stable foreground control surface on Linux.
        # This avoids silent exits when tray backend terminates unexpectedly.
        icon.run_detached()
        self._run_fallback_window(do_auto_open=self._should_open_controls_on_start())

    def _should_open_controls_on_start(self) -> bool:
        session = str(os.environ.get("XDG_SESSION_TYPE", "")).strip().lower()
        desktop = str(os.environ.get("XDG_CURRENT_DESKTOP", "")).strip().lower()
        return session == "wayland" and "gnome" in desktop

    def _open_control_window(self, *_) -> None:
        """Open/raise the control window (left-click default action)."""
        root = self._control_root
        if root is not None:
            try:
                # Schedule on the tk main thread to avoid cross-thread issues.
                root.after(0, lambda: (root.deiconify(), root.lift()))
            except Exception:
                pass
            return
        threading.Thread(target=self._run_fallback_window, kwargs={"do_auto_open": True}, daemon=True).start()

    def _on_state_changed(self, state: ClientState) -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = self._make_icon(state)
            status = "connected" if state.connected else "disconnected"
            self._last_status_line = (
                f"OpenClaw Voice ({status}) | tts_muted={state.tts_muted} | "
                f"continuous={state.continuous_mode} | rms={state.mic_rms:.2f}"
            )
            icon.title = self._last_status_line
            icon.update_menu()
        except Exception:
            # Some Linux tray backends can raise on cross-thread or transient redraw calls.
            # Keep the app alive and continue serving menu actions.
            pass

    def _run_fallback_window(self, do_auto_open: bool = True) -> None:
        import tkinter as tk

        self._fallback_opened = True
        root = tk.Tk()
        self._control_root = root
        root.title("OpenClaw Voice Controls")
        root.geometry("360x220")
        root.resizable(False, False)

        status_var = tk.StringVar(value=self._last_status_line)

        frame = tk.Frame(root, padx=14, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            frame,
            text="Tray icon unavailable in this session.\nUse this control window.",
            justify="left",
            anchor="w",
        ).pack(anchor="w")

        tk.Label(frame, textvariable=status_var, justify="left", wraplength=330, anchor="w").pack(anchor="w", pady=(10, 10))

        row1 = tk.Frame(frame)
        row1.pack(fill=tk.X, pady=4)
        tk.Button(row1, text="Toggle Microphone", command=self._toggle_mic, width=18).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(row1, text="Open Web UI", command=self._open_web_ui, width=14).pack(side=tk.LEFT)

        row2 = tk.Frame(frame)
        row2.pack(fill=tk.X, pady=4)
        tk.Button(row2, text="Mute TTS", command=self._toggle_tts, width=18).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(row2, text="Continuous Mode", command=self._toggle_continuous, width=14).pack(side=tk.LEFT)

        row3 = tk.Frame(frame)
        row3.pack(fill=tk.X, pady=4)
        tk.Button(row3, text="Settings", command=self._open_settings, width=18).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(row3, text="Quit", command=root.destroy, width=14).pack(side=tk.LEFT)

        def _refresh_status() -> None:
            status_var.set(self._last_status_line)
            root.after(500, _refresh_status)

        root.after(500, _refresh_status)
        if not do_auto_open:
            root.withdraw()  # hidden but event loop keeps app alive; deiconify on left-click
        root.mainloop()
        self._fallback_opened = False
        self._control_root = None

    def _make_icon(self, state: ClientState):
        Image, ImageDraw = self._pil_image
        size = 64
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)

        stroke = border_width_for_state(state.connected, state.mic_enabled, state.mic_rms)
        color = (130, 130, 130, 255)
        if state.connected:
            color = (220, 60, 60, 255)
            if state.wake_state == "awake":
                color = (50, 185, 95, 255)

        W = (242, 242, 246, 255)  # near-white glyph colour

        # ── Dark background disk ────────────────────────────────────────────────
        d.ellipse((6, 6, 58, 58), fill=(36, 36, 46, 255))

        # ── Coloured VU ring ────────────────────────────────────────────────────
        d.ellipse((6, 6, 58, 58), outline=color, width=max(2, stroke))

        # ── Mic capsule body (pill shape built from primitives) ─────────────────
        # Capsule: x 26-38, y 10-34  (12 wide, 24 tall)
        cx, cy_top, cy_bot = 32, 10, 34
        cap_l, cap_r = 26, 38
        d.rectangle((cap_l, cy_top + 6, cap_r, cy_bot), fill=W)   # body rect
        d.ellipse((cap_l, cy_top, cap_r, cy_top + 12), fill=W)     # top dome

        # ── Horizontal stand bar ────────────────────────────────────────────────
        d.rectangle((23, 48, 41, 51), fill=W)

        # ── Vertical stem (centre) ──────────────────────────────────────────────
        d.rectangle((30, 38, 34, 49), fill=W)

        # ── U-shaped arm (arc below capsule) ────────────────────────────────────
        # Arc bounding box: left of the capsule to right of capsule, height ~16
        d.arc((20, 32, 44, 48), start=0, end=180, fill=W, width=3)

        # ── Strike-through diagonal when disconnected ───────────────────────────
        if not state.connected:
            d.line((16, 48, 48, 16), fill=(240, 200, 80, 255), width=4)

        return im

    def _toggle_mic(self, *_: Any) -> None:
        self.controller.trigger_mic_toggle()

    def _open_web_ui(self, *_: Any) -> None:
        self.controller.open_web_ui()

    def _toggle_tts(self, *_: Any) -> None:
        self.controller.toggle_tts_mute()

    def _toggle_continuous(self, *_: Any) -> None:
        self.controller.toggle_continuous_mode()

    def _open_settings(self, *_: Any) -> None:
        cfg = self.controller.config
        initial = {
            "DESKTOP_WEB_UI_URL": cfg.web_ui_url,
        }

        def on_save(values: dict[str, str]) -> None:
            cfg.web_ui_url = values.get("DESKTOP_WEB_UI_URL", cfg.web_ui_url)
            cfg.ws_url = derive_ws_url(cfg.web_ui_url)

            write_desktop_env(
                cfg.desktop_env_path,
                {
                    "DESKTOP_WEB_UI_URL": cfg.web_ui_url,
                    "DESKTOP_DEFAULT_TTS_MUTED": str(self.controller.state.tts_muted).lower(),
                    "DESKTOP_DEFAULT_CONTINUOUS_MODE": str(self.controller.state.continuous_mode).lower(),
                    "DESKTOP_RECONNECT_DELAY_S": str(cfg.reconnect_delay_s),
                },
            )

        threading.Thread(target=lambda: SettingsDialog(initial, on_save).show(), daemon=True).start()

    def _quit(self, *_: Any) -> None:
        if self._icon is not None:
            self._icon.stop()
