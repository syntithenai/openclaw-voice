from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_TRACE_ENABLED = str(os.getenv("OPENCLAW_MUSIC_LATENCY_TRACE", "")).strip().lower() in {"1", "true", "yes", "on"}
_TRACE_LOCK = threading.Lock()
_TRACE_PATH: Path | None = None
_ACTION_ID_CTX: ContextVar[str] = ContextVar("music_latency_action_id", default="")


def is_enabled() -> bool:
    return _TRACE_ENABLED


def current_action_id() -> str:
    return str(_ACTION_ID_CTX.get() or "")


@contextmanager
def scoped_action(action_id: str):
    token = _ACTION_ID_CTX.set(str(action_id or ""))
    try:
        yield
    finally:
        _ACTION_ID_CTX.reset(token)


def _workspace_dir() -> Path:
    configured = str(os.getenv("OPENCLAW_WORKSPACE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".openclaw").resolve()


def _run_id() -> str:
    configured = str(os.getenv("OPENCLAW_MUSIC_LATENCY_RUN_ID", "")).strip()
    if configured:
        return configured
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _trace_path() -> Path:
    global _TRACE_PATH
    if _TRACE_PATH is not None:
        return _TRACE_PATH
    base = _workspace_dir() / "benchmarks" / "playlist-load"
    base.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH = base / f"server-trace-{_run_id()}.jsonl"
    return _TRACE_PATH


def emit(event: str, action_id: str = "", **fields: Any) -> None:
    if not _TRACE_ENABLED:
        return
    action = str(action_id or "").strip() or current_action_id()
    record: dict[str, Any] = {
        "event": str(event),
        "action_id": action,
        "ts": time.time(),
        "mono": time.monotonic(),
    }
    if fields:
        record.update(fields)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    path = _trace_path()
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
