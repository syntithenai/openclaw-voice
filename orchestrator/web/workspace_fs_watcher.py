from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Awaitable, Callable

from inotify_simple import INotify, flags


class WorkspaceFsWatcher:
    def __init__(
        self,
        root: Path,
        excluded_dirs: set[str],
        max_watches: int,
        max_events_per_tick: int,
        max_paths_per_push: int,
        coalesce_ms: int,
        on_change: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        self.root = root.resolve()
        self.excluded_dirs = set(excluded_dirs or set())
        self.max_watches = max(64, int(max_watches))
        self.max_events_per_tick = max(16, int(max_events_per_tick))
        self.max_paths_per_push = max(8, int(max_paths_per_push))
        self.coalesce_s = max(0.01, int(coalesce_ms) / 1000.0)
        self.on_change = on_change

        self._inotify: INotify | None = None
        self._task: asyncio.Task[None] | None = None
        self._wd_to_path: dict[int, Path] = {}

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._inotify = INotify()
        try:
            self._add_recursive(self.root)
        except Exception:
            await self._emit_resync("limit")
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        if self._inotify is not None:
            with contextlib.suppress(Exception):
                self._inotify.close()
        self._inotify = None
        self._wd_to_path.clear()

    def _is_excluded(self, path: Path) -> bool:
        if path == self.root:
            return False
        return any(part in self.excluded_dirs for part in path.parts)

    def _add_watch(self, path: Path) -> None:
        if self._inotify is None:
            raise RuntimeError("watcher not initialized")
        if len(self._wd_to_path) >= self.max_watches:
            raise RuntimeError("inotify watch limit reached")
        if self._is_excluded(path.relative_to(self.root)):
            return
        mask = (
            flags.CREATE
            | flags.DELETE
            | flags.MODIFY
            | flags.MOVED_FROM
            | flags.MOVED_TO
            | flags.CLOSE_WRITE
            | flags.ATTRIB
            | flags.DELETE_SELF
            | flags.MOVE_SELF
            | flags.Q_OVERFLOW
        )
        wd = self._inotify.add_watch(str(path), mask)
        self._wd_to_path[wd] = path

    def _add_recursive(self, root: Path) -> None:
        if not root.exists() or not root.is_dir():
            return
        self._add_watch(root)
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            rel = p.relative_to(self.root)
            if self._is_excluded(rel):
                continue
            self._add_watch(p)

    def _to_api_path(self, candidate: Path) -> str | None:
        try:
            rel = candidate.relative_to(self.root)
        except ValueError:
            return None
        if self._is_excluded(rel):
            return None
        if str(rel) == ".":
            return "/"
        return "/" + str(rel).replace("\\", "/")

    async def _emit_resync(self, reason: str) -> None:
        await self.on_change({"reason": reason, "resyncRequired": True, "paths": []})

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.coalesce_s)
            if self._inotify is None:
                continue
            events = await asyncio.to_thread(self._inotify.read, 0)
            if not events:
                continue

            if len(events) > self.max_events_per_tick:
                await self._emit_resync("limit")
                continue

            changed: list[str] = []
            overflow = False

            for ev in events:
                ev_flags = flags.from_mask(ev.mask)
                if flags.Q_OVERFLOW in ev_flags:
                    overflow = True
                    break

                base = self._wd_to_path.get(ev.wd)
                if base is None:
                    continue

                ev_name = str(ev.name or "").strip()
                candidate = (base / ev_name).resolve() if ev_name else base.resolve()

                if self.root != candidate and self.root not in candidate.parents:
                    continue

                if flags.ISDIR in ev_flags and flags.CREATE in ev_flags:
                    with contextlib.suppress(Exception):
                        self._add_watch(candidate)

                if flags.DELETE_SELF in ev_flags or flags.MOVE_SELF in ev_flags:
                    with contextlib.suppress(KeyError):
                        self._wd_to_path.pop(ev.wd)

                api_path = self._to_api_path(candidate)
                if api_path:
                    changed.append(api_path)
                if len(changed) >= self.max_paths_per_push:
                    break

            if overflow:
                await self._emit_resync("overflow")
                continue

            if len(changed) >= self.max_paths_per_push:
                await self._emit_resync("limit")
                continue

            unique = sorted(set(changed))
            if unique:
                await self.on_change({"reason": "changed", "resyncRequired": False, "paths": unique})
