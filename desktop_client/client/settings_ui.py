from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from urllib.parse import urlparse


def _is_http_url(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_settings(values: dict[str, str]) -> tuple[bool, str]:
    web_ui = str(values.get("DESKTOP_WEB_UI_URL", "")).strip()

    if not web_ui:
        return False, "Web UI URL is required."
    if not _is_http_url(web_ui):
        return False, "Web UI URL must be a valid http/https URL."
    return True, ""


class SettingsDialog:
    def __init__(self, initial_values: dict[str, str], on_save) -> None:
        self._initial_values = dict(initial_values)
        self._on_save = on_save
        self._root: tk.Tk | None = None
        self._entries: dict[str, tk.Entry] = {}

    def show(self) -> None:
        root = tk.Tk()
        root.title("OpenClaw Desktop Client Settings")
        root.geometry("640x160")
        root.resizable(False, False)
        self._root = root

        fields = [
            ("DESKTOP_WEB_UI_URL", "OpenClaw Voice Web UI URL"),
        ]

        frame = tk.Frame(root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        for row, (key, label) in enumerate(fields):
            tk.Label(frame, text=label, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
            entry = tk.Entry(frame, width=68)
            entry.insert(0, str(self._initial_values.get(key, "")))
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self._entries[key] = entry

        frame.grid_columnconfigure(1, weight=1)

        button_row = tk.Frame(frame)
        button_row.grid(row=len(fields), column=0, columnspan=2, sticky="e", pady=(14, 0))

        tk.Button(button_row, text="Cancel", command=root.destroy).pack(side=tk.RIGHT, padx=6)
        tk.Button(button_row, text="Save", command=self._save).pack(side=tk.RIGHT)

        root.mainloop()

    def _save(self) -> None:
        assert self._root is not None
        values = {k: e.get().strip() for k, e in self._entries.items()}
        ok, message = validate_settings(values)
        if not ok:
            messagebox.showerror("Invalid settings", message)
            return
        self._on_save(values)
        self._root.destroy()
