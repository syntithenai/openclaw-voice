# Workspace File Manager Plan (Implementation Ready)

## Final Recommendations (Locked)
The implementation will use:
- Markdown editor: EasyMDE (MIT, maintained, vanilla JS, drop-in for textarea).
- Expandable JSON editor: vanilla-jsoneditor (successor to jsoneditor, modern tree/text/table, vanilla JS support).

These choices optimize for this codebase because the embedded web UI is plain JS and not framework-based.

## Scope
Build a web UI file manager in the existing embedded UI stack (vanilla JS + embedded Python HTTP service) with:
- Config-based exclusion of folders: recordings, playlists, timers, .media, .openclaw.
- Expandable file tree on the left.
- Folder content list in the center.
- Right pane editor/preview with type routing:
  - markdown -> EasyMDE
  - json -> vanilla-jsoneditor tree mode
  - known text files -> plain textarea
  - non-text -> read-only preview/info, including media previews
- Debounced autosave.
- Create folder action for selected folder.
- Top-level synthetic folder "OpenClaw Configuration" containing top-level files:
  - SOUL.md, BOOTSTRAP.md, TOOLS.md, HEARTBEAT.md, IDENTITY.md, USER.md, AGENTS.md
- Orchestrator-side inotify watcher (strict limits) that propagates filesystem updates immediately to active UI clients.
- File manager menu placement after Recordings in both desktop and mobile navigation.

## Key Pain Points and Concrete Solutions

### 1) Path safety and exclusion policy
- Centralize all path normalization and root checks in a dedicated backend service.
- Enforce exclusions in backend API responses and writes.
- Reject path traversal, invalid virtual paths, and excluded target names.

### 2) Tree performance
- Lazy-load one directory level at a time.
- Cache children by path in frontend state.
- Invalidate folder cache on create/save operations.

### 3) Editor routing and reliability
- Backend returns category + mime + editable flags.
- Frontend mounts editor by category with adapter-style helper functions.
- Keep markdown/json adapter boundaries explicit so editor replacement is easy.

### 4) Autosave conflicts
- Debounce per-file writes (500 ms default).
- Use optimistic concurrency token (etag) from load to save.
- Return 409 conflict on stale write; UI can show conflict and reload prompt.

### 5) Synthetic top-level config folder
- Hide listed top-level config files from root listing.
- Inject synthetic folder node.
- Map virtual paths to actual top-level files in backend service.

### 6) Live filesystem synchronization (inotify, strict limits)
- Use inotify_simple in the orchestrator for workspace file manager roots.
- Watch recursively with hard guardrails:
  - max active watches
  - max events processed per tick
  - max paths emitted per websocket payload
  - short coalescing window to collapse bursts
- On overflow or guardrail breach, emit an immediate resync-required event so the UI performs a full refresh.
- Push change events to active UIs over websocket immediately; never depend on periodic polling for freshness.

### 7) Navigation placement consistency
- Ensure Files is rendered after Recordings in both desktop and mobile menus.
- Keep hash route #/files and active-nav highlighting behavior consistent with existing pages.

## API Contract

### Endpoints
- GET /api/file-manager/tree?path=/...
- GET /api/file-manager/folder?path=/...
- GET /api/file-manager/file?path=/...
- GET /api/file-manager/preview?path=/...
- PUT /api/file-manager/file?path=/...
- POST /api/file-manager/folder?path=/...

### Response Shapes
- Tree/folder item entry:
  - name, path, kind (folder|file|virtual-folder), isVirtual, editable, mimeType, size, mtime
- File detail:
  - path, name, category (markdown|json|text|media|binary), mimeType, size, mtime, etag, editable, readOnlyReason, content, previewUrl

### WebSocket Event
- file_manager_fs_changed
  - rev: monotonic sequence number
  - paths: changed API paths (bounded by strict limit)
  - reason: changed|overflow|limit
  - resyncRequired: boolean

## Implementation Code Pack
The following code is the complete implementation blueprint to apply.

---

### 1) New Backend Service

File: orchestrator/web/file_manager_service.py

```python
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any


class FileManagerError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class WorkspaceFileManager:
    VIRTUAL_CONFIG_ROOT = "/__virtual__/openclaw-config"

    MARKDOWN_EXTS = {".md", ".markdown", ".mdown", ".mkd"}
    JSON_EXTS = {".json", ".jsonc"}
    TEXT_EXTS = {
        ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".scss", ".html", ".htm",
        ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".zsh",
        ".md", ".sql", ".env", ".gitignore", ".dockerignore", ".log", ".csv",
    }
    MEDIA_EXTS = {
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
        ".mp4", ".mkv", ".mov", ".webm",
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    }

    def __init__(
        self,
        root: str,
        excluded_folders: list[str] | set[str],
        excluded_top_level_config_files: list[str] | set[str],
        max_editable_bytes: int = 2_000_000,
    ):
        self.root = Path(root).expanduser().resolve()
        self.excluded_folders = {str(x).strip() for x in excluded_folders if str(x).strip()}
        self.excluded_top_level_config_files = {
            str(x).strip() for x in excluded_top_level_config_files if str(x).strip()
        }
        self.max_editable_bytes = max(64_000, int(max_editable_bytes or 2_000_000))

        self.root.mkdir(parents=True, exist_ok=True)

    def _normalize_api_path(self, value: str | None) -> str:
        raw = str(value or "/").strip()
        if not raw:
            raw = "/"
        if not raw.startswith("/"):
            raw = "/" + raw
        if raw != "/" and raw.endswith("/"):
            raw = raw[:-1]
        return raw

    def _is_virtual(self, api_path: str) -> bool:
        return api_path == self.VIRTUAL_CONFIG_ROOT or api_path.startswith(self.VIRTUAL_CONFIG_ROOT + "/")

    def _resolve_real(self, api_path: str) -> Path:
        normalized = self._normalize_api_path(api_path)
        rel = normalized.lstrip("/")
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise FileManagerError(400, "path escapes workspace root")
        return candidate

    def _resolve_virtual_file(self, api_path: str) -> Path:
        normalized = self._normalize_api_path(api_path)
        if not normalized.startswith(self.VIRTUAL_CONFIG_ROOT + "/"):
            raise FileManagerError(400, "invalid virtual file path")
        file_name = normalized.split("/")[-1]
        if file_name not in self.excluded_top_level_config_files:
            raise FileManagerError(404, "virtual file not found")
        real = (self.root / file_name).resolve()
        if real != self.root and self.root not in real.parents:
            raise FileManagerError(400, "invalid virtual mapping")
        return real

    def _etag_for_path(self, path: Path) -> str:
        st = path.stat()
        return f"{int(st.st_mtime_ns)}:{int(st.st_size)}"

    def _looks_binary(self, sample: bytes) -> bool:
        if not sample:
            return False
        if b"\x00" in sample:
            return True
        bad = 0
        for b in sample:
            if b in (9, 10, 13):
                continue
            if b < 32:
                bad += 1
        return bad > max(6, int(len(sample) * 0.25))

    def _categorize(self, path: Path, mime_type: str) -> str:
        ext = path.suffix.lower()
        if ext in self.MARKDOWN_EXTS:
            return "markdown"
        if ext in self.JSON_EXTS:
            return "json"
        if ext in self.MEDIA_EXTS:
            return "media"
        if mime_type.startswith("audio/") or mime_type.startswith("video/") or mime_type.startswith("image/"):
            return "media"
        if ext in self.TEXT_EXTS or mime_type.startswith("text/"):
            return "text"
        try:
            with path.open("rb") as handle:
                sample = handle.read(1024)
        except Exception:
            return "binary"
        return "binary" if self._looks_binary(sample) else "text"

    def _entry(self, path: Path, *, is_virtual: bool = False, virtual_path: str = "") -> dict[str, Any]:
        if is_virtual:
            return {
                "name": "OpenClaw Configuration",
                "path": self.VIRTUAL_CONFIG_ROOT,
                "kind": "virtual-folder",
                "isVirtual": True,
                "editable": False,
                "mimeType": "",
                "size": 0,
                "mtime": 0,
            }

        st = path.stat()
        rel = "/" + str(path.relative_to(self.root)).replace("\\", "/")
        if path.is_dir():
            return {
                "name": path.name,
                "path": rel,
                "kind": "folder",
                "isVirtual": False,
                "editable": False,
                "mimeType": "inode/directory",
                "size": 0,
                "mtime": int(st.st_mtime),
            }

        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        category = self._categorize(path, mime_type)
        editable = category in {"markdown", "json", "text"} and st.st_size <= self.max_editable_bytes
        return {
            "name": path.name,
            "path": virtual_path or rel,
            "kind": "file",
            "isVirtual": bool(virtual_path),
            "editable": editable,
            "mimeType": mime_type,
            "size": int(st.st_size),
            "mtime": int(st.st_mtime),
        }

    def _list_children(self, folder: Path, *, top_level: bool) -> list[dict[str, Any]]:
        children: list[Path] = []
        try:
            children = list(folder.iterdir())
        except Exception as exc:
            raise FileManagerError(500, f"failed to list directory: {exc}") from exc

        items: list[dict[str, Any]] = []
        for child in children:
            if child.is_dir() and child.name in self.excluded_folders:
                continue
            if top_level and child.is_file() and child.name in self.excluded_top_level_config_files:
                continue
            items.append(self._entry(child))

        items.sort(key=lambda item: (item["kind"] not in {"folder", "virtual-folder"}, str(item["name"]).lower()))
        return items

    def list_tree(self, path: str | None) -> dict[str, Any]:
        api_path = self._normalize_api_path(path)
        if api_path == self.VIRTUAL_CONFIG_ROOT:
            children: list[dict[str, Any]] = []
            for name in sorted(self.excluded_top_level_config_files, key=lambda x: x.lower()):
                real = (self.root / name).resolve()
                if not real.exists() or not real.is_file():
                    continue
                virtual_path = f"{self.VIRTUAL_CONFIG_ROOT}/{name}"
                children.append(self._entry(real, virtual_path=virtual_path))
            return {"path": api_path, "children": children}

        real = self._resolve_real(api_path)
        if not real.exists() or not real.is_dir():
            raise FileManagerError(404, "folder not found")

        top_level = real == self.root
        children = self._list_children(real, top_level=top_level)

        if top_level:
            has_config_files = any((self.root / f).exists() for f in self.excluded_top_level_config_files)
            if has_config_files:
                children.insert(0, self._entry(self.root, is_virtual=True))

        return {"path": api_path, "children": children}

    def list_folder(self, path: str | None) -> dict[str, Any]:
        return self.list_tree(path)

    def get_file(self, path: str | None) -> dict[str, Any]:
        api_path = self._normalize_api_path(path)
        real = self._resolve_virtual_file(api_path) if self._is_virtual(api_path) else self._resolve_real(api_path)

        if not real.exists() or not real.is_file():
            raise FileManagerError(404, "file not found")

        st = real.stat()
        mime_type = mimetypes.guess_type(str(real))[0] or "application/octet-stream"
        category = self._categorize(real, mime_type)
        editable = category in {"markdown", "json", "text"} and st.st_size <= self.max_editable_bytes
        read_only_reason = ""
        if category in {"markdown", "json", "text"} and st.st_size > self.max_editable_bytes:
            read_only_reason = f"file exceeds max editable size ({self.max_editable_bytes} bytes)"

        content = ""
        if editable:
            try:
                content = real.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                raise FileManagerError(500, f"failed reading file: {exc}") from exc

        preview_url = ""
        if category in {"media", "binary"}:
            preview_url = f"/api/file-manager/preview?path={api_path}"

        return {
            "path": api_path,
            "name": real.name,
            "category": category,
            "mimeType": mime_type,
            "size": int(st.st_size),
            "mtime": int(st.st_mtime),
            "etag": self._etag_for_path(real),
            "editable": editable,
            "readOnlyReason": read_only_reason,
            "content": content,
            "previewUrl": preview_url,
        }

    def save_file(self, path: str | None, content: str, expected_etag: str = "") -> dict[str, Any]:
        api_path = self._normalize_api_path(path)
        real = self._resolve_virtual_file(api_path) if self._is_virtual(api_path) else self._resolve_real(api_path)

        if not real.exists() or not real.is_file():
            raise FileManagerError(404, "file not found")

        mime_type = mimetypes.guess_type(str(real))[0] or "application/octet-stream"
        category = self._categorize(real, mime_type)
        if category not in {"markdown", "json", "text"}:
            raise FileManagerError(400, "file is not editable")

        if expected_etag and expected_etag != self._etag_for_path(real):
            raise FileManagerError(409, "file changed on disk")

        encoded = content.encode("utf-8")
        if len(encoded) > self.max_editable_bytes:
            raise FileManagerError(413, f"content exceeds max editable size ({self.max_editable_bytes} bytes)")

        try:
            real.write_text(content, encoding="utf-8")
        except Exception as exc:
            raise FileManagerError(500, f"failed saving file: {exc}") from exc

        st = real.stat()
        return {
            "path": api_path,
            "etag": self._etag_for_path(real),
            "size": int(st.st_size),
            "mtime": int(st.st_mtime),
        }

    def create_folder(self, parent_path: str | None, name: str) -> dict[str, Any]:
        api_parent = self._normalize_api_path(parent_path)
        if self._is_virtual(api_parent):
            raise FileManagerError(400, "cannot create folders under virtual root")

        folder_name = str(name or "").strip()
        if not folder_name:
            raise FileManagerError(400, "folder name is required")
        if "/" in folder_name or "\\" in folder_name:
            raise FileManagerError(400, "folder name must not include path separators")
        if folder_name in {".", ".."}:
            raise FileManagerError(400, "folder name is invalid")
        if folder_name in self.excluded_folders:
            raise FileManagerError(400, "folder name is excluded by configuration")

        parent = self._resolve_real(api_parent)
        if not parent.exists() or not parent.is_dir():
            raise FileManagerError(404, "parent folder not found")

        new_folder = (parent / folder_name).resolve()
        if new_folder != parent and parent not in new_folder.parents:
            raise FileManagerError(400, "invalid target folder")
        if new_folder.exists():
            raise FileManagerError(409, "folder already exists")

        try:
            new_folder.mkdir(parents=False, exist_ok=False)
        except Exception as exc:
            raise FileManagerError(500, f"failed creating folder: {exc}") from exc

        return {"entry": self._entry(new_folder)}

    def resolve_preview_path(self, path: str | None) -> Path:
        api_path = self._normalize_api_path(path)
        real = self._resolve_virtual_file(api_path) if self._is_virtual(api_path) else self._resolve_real(api_path)
        if not real.exists() or not real.is_file():
            raise FileManagerError(404, "file not found")
        return real
```

---

### 2) Config Fields

File: orchestrator/config.py

Add these fields to VoiceConfig near existing web_ui_* settings:

```python
    web_ui_file_manager_enabled: bool = Field(True)  # Enable file manager page + APIs
    web_ui_file_manager_root: str = Field("")  # Empty = OPENCLAW_WORKSPACE_DIR
    web_ui_file_manager_excluded_folders: str = Field("recordings,playlists,timers,.media,.openclaw")
    web_ui_file_manager_top_level_config_files: str = Field("SOUL.md,BOOTSTRAP.md,TOOLS.md,HEARTBEAT.md,IDENTITY.md,USER.md,AGENTS.md")
    web_ui_file_manager_max_editable_bytes: int = Field(2_000_000)
  web_ui_file_manager_watch_enabled: bool = Field(True)
  web_ui_file_manager_watch_max_watches: int = Field(4096)
  web_ui_file_manager_watch_max_events_per_tick: int = Field(256)
  web_ui_file_manager_watch_max_paths_per_push: int = Field(128)
  web_ui_file_manager_watch_coalesce_ms: int = Field(75)
```

---

### 3) Environment Template

File: .env.example

Add:

```env
WEB_UI_FILE_MANAGER_ENABLED=true
WEB_UI_FILE_MANAGER_ROOT=
WEB_UI_FILE_MANAGER_EXCLUDED_FOLDERS=recordings,playlists,timers,.media,.openclaw
WEB_UI_FILE_MANAGER_TOP_LEVEL_CONFIG_FILES=SOUL.md,BOOTSTRAP.md,TOOLS.md,HEARTBEAT.md,IDENTITY.md,USER.md,AGENTS.md
WEB_UI_FILE_MANAGER_MAX_EDITABLE_BYTES=2000000
WEB_UI_FILE_MANAGER_WATCH_ENABLED=true
WEB_UI_FILE_MANAGER_WATCH_MAX_WATCHES=4096
WEB_UI_FILE_MANAGER_WATCH_MAX_EVENTS_PER_TICK=256
WEB_UI_FILE_MANAGER_WATCH_MAX_PATHS_PER_PUSH=128
WEB_UI_FILE_MANAGER_WATCH_COALESCE_MS=75
```

---

### 4) Service Wiring from Main

File: orchestrator/main.py

Extend EmbeddedVoiceWebService constructor call:

```python
                file_manager_enabled=config.web_ui_file_manager_enabled,
                file_manager_root=config.web_ui_file_manager_root,
                file_manager_excluded_folders=config.web_ui_file_manager_excluded_folders,
                file_manager_top_level_config_files=config.web_ui_file_manager_top_level_config_files,
                file_manager_max_editable_bytes=config.web_ui_file_manager_max_editable_bytes,
                file_manager_watch_enabled=config.web_ui_file_manager_watch_enabled,
                file_manager_watch_max_watches=config.web_ui_file_manager_watch_max_watches,
                file_manager_watch_max_events_per_tick=config.web_ui_file_manager_watch_max_events_per_tick,
                file_manager_watch_max_paths_per_push=config.web_ui_file_manager_watch_max_paths_per_push,
                file_manager_watch_coalesce_ms=config.web_ui_file_manager_watch_coalesce_ms,
```

---

### 5) Initialize File Manager in Realtime Service

File: orchestrator/web/realtime_service.py

Add import:

```python
from orchestrator.web.file_manager_service import WorkspaceFileManager
from orchestrator.web.workspace_fs_watcher import WorkspaceFsWatcher
```

Extend __init__ signature:

```python
        file_manager_enabled: bool = False,
        file_manager_root: str = "",
        file_manager_excluded_folders: str = "recordings,playlists,timers,.media,.openclaw",
        file_manager_top_level_config_files: str = "SOUL.md,BOOTSTRAP.md,TOOLS.md,HEARTBEAT.md,IDENTITY.md,USER.md,AGENTS.md",
        file_manager_max_editable_bytes: int = 2_000_000,
  file_manager_watch_enabled: bool = True,
  file_manager_watch_max_watches: int = 4096,
  file_manager_watch_max_events_per_tick: int = 256,
  file_manager_watch_max_paths_per_push: int = 128,
  file_manager_watch_coalesce_ms: int = 75,
```

Add initialization near workspace/media roots:

```python
        fm_root = Path(file_manager_root).expanduser() if file_manager_root else workspace_root
        self.file_manager_enabled = bool(file_manager_enabled)
        self.file_manager = WorkspaceFileManager(
            root=str(fm_root.resolve()),
            excluded_folders=[x.strip() for x in str(file_manager_excluded_folders or "").split(",") if x.strip()],
            excluded_top_level_config_files=[x.strip() for x in str(file_manager_top_level_config_files or "").split(",") if x.strip()],
            max_editable_bytes=int(file_manager_max_editable_bytes or 2_000_000),
        )

    self._file_manager_fs_rev = 0
    self._file_manager_watcher: WorkspaceFsWatcher | None = None
    if self.file_manager_enabled and file_manager_watch_enabled:
      self._file_manager_watcher = WorkspaceFsWatcher(
        root=Path(self.file_manager.root),
        excluded_dirs=self.file_manager.excluded_folders,
        max_watches=int(file_manager_watch_max_watches),
        max_events_per_tick=int(file_manager_watch_max_events_per_tick),
        max_paths_per_push=int(file_manager_watch_max_paths_per_push),
        coalesce_ms=int(file_manager_watch_coalesce_ms),
        on_change=self._on_file_manager_fs_change,
      )
```

Add realtime callback + websocket push:

```python
  async def _on_file_manager_fs_change(self, payload: dict[str, Any]) -> None:
    self._file_manager_fs_rev += 1
    await self._broadcast_json(
      {
        "type": "file_manager_fs_changed",
        "rev": self._file_manager_fs_rev,
        "paths": payload.get("paths", []),
        "reason": payload.get("reason", "changed"),
        "resyncRequired": bool(payload.get("resyncRequired", False)),
      }
    )
```

Lifecycle hooks:

```python
    if self._file_manager_watcher:
      await self._file_manager_watcher.start()
```

```python
    if self._file_manager_watcher:
      await self._file_manager_watcher.stop()
```

New watcher module:

File: orchestrator/web/workspace_fs_watcher.py

```python
from __future__ import annotations

import asyncio
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
    self._task: asyncio.Task | None = None
    self._wd_to_path: dict[int, Path] = {}

  async def start(self) -> None:
    self._inotify = INotify()
    self._add_recursive(self.root)
    self._task = asyncio.create_task(self._loop())

  async def stop(self) -> None:
    if self._task:
      self._task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._task
    self._task = None

  def _add_watch(self, path: Path) -> None:
    if len(self._wd_to_path) >= self.max_watches:
      raise RuntimeError("inotify watch limit reached")
    mask = (
      flags.CREATE | flags.DELETE | flags.MODIFY | flags.MOVED_FROM | flags.MOVED_TO |
      flags.CLOSE_WRITE | flags.ATTRIB | flags.DELETE_SELF | flags.MOVE_SELF | flags.Q_OVERFLOW
    )
    wd = self._inotify.add_watch(str(path), mask)
    self._wd_to_path[wd] = path

  def _add_recursive(self, root: Path) -> None:
    for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
      if p.name in self.excluded_dirs:
        continue
      self._add_watch(p)

  async def _loop(self) -> None:
    while True:
      await asyncio.sleep(self.coalesce_s)
      events = self._inotify.read(timeout=0)
      if not events:
        continue
      if len(events) > self.max_events_per_tick:
        await self.on_change({"reason": "limit", "resyncRequired": True, "paths": []})
        continue

      changed: list[str] = []
      overflow = False
      for ev in events:
        if flags.Q_OVERFLOW in flags.from_mask(ev.mask):
          overflow = True
          break
        base = self._wd_to_path.get(ev.wd)
        if not base:
          continue
        candidate = (base / ev.name).resolve() if ev.name else base.resolve()
        if self.root != candidate and self.root not in candidate.parents:
          continue
        rel = "/" + str(candidate.relative_to(self.root)).replace("\\", "/")
        changed.append(rel)
        if len(changed) >= self.max_paths_per_push:
          break

      if overflow:
        await self.on_change({"reason": "overflow", "resyncRequired": True, "paths": []})
      elif changed:
        await self.on_change({"reason": "changed", "resyncRequired": False, "paths": sorted(set(changed))})
```

Update should_protect_http_path so auth-required mode also protects file manager APIs:

```python
        protected_prefixes = [
            "/files/workspace",
            "/files/media",
            "/recordings/audio/",
            "/api/file-manager",
        ]
```

---

### 6) Add File Manager API Routes

File: orchestrator/web/http_server.py

Add import:

```python
from orchestrator.web.file_manager_service import FileManagerError
```

Inside UIHandler add helpers:

```python
        def _require_file_manager(self):
            manager = getattr(service, "file_manager", None)
            enabled = bool(getattr(service, "file_manager_enabled", False))
            if not enabled or manager is None:
                raise FileManagerError(404, "file manager is disabled")
            return manager

        def _query_path(self, query: dict[str, list[str]], default: str = "/") -> str:
            return str((query.get("path") or [default])[0] or default)

        def _handle_file_manager_get(self, path: str, query: dict[str, list[str]]) -> bool:
            if not path.startswith("/api/file-manager"):
                return False
            try:
                manager = self._require_file_manager()
                if path == "/api/file-manager/tree":
                    self._send_json(manager.list_tree(self._query_path(query, "/")))
                    return True
                if path == "/api/file-manager/folder":
                    self._send_json(manager.list_folder(self._query_path(query, "/")))
                    return True
                if path == "/api/file-manager/file":
                    self._send_json(manager.get_file(self._query_path(query, "/")))
                    return True
                if path == "/api/file-manager/preview":
                    preview_path = manager.resolve_preview_path(self._query_path(query, "/"))
                    self._send_file(preview_path)
                    return True
                self._send_json({"error": "Not found"}, status=404)
                return True
            except FileManagerError as exc:
                self._send_json({"error": exc.message}, status=exc.status)
                return True

        def _handle_file_manager_post(self, path: str, query: dict[str, list[str]], body: bytes) -> bool:
            if path != "/api/file-manager/folder":
                return False
            try:
                manager = self._require_file_manager()
                payload = json.loads(body.decode("utf-8") or "{}")
                name = str(payload.get("name", ""))
                parent = self._query_path(query, "/")
                self._send_json(manager.create_folder(parent, name))
                return True
            except json.JSONDecodeError:
                self._send_json({"error": "invalid request body"}, status=400)
                return True
            except FileManagerError as exc:
                self._send_json({"error": exc.message}, status=exc.status)
                return True

        def _handle_file_manager_put(self, path: str, query: dict[str, list[str]], body: bytes) -> bool:
            if path != "/api/file-manager/file":
                return False
            try:
                manager = self._require_file_manager()
                payload = json.loads(body.decode("utf-8") or "{}")
                content = str(payload.get("content", ""))
                expected_etag = str(payload.get("expectedEtag", ""))
                file_path = self._query_path(query, "/")
                self._send_json(manager.save_file(file_path, content, expected_etag))
                return True
            except json.JSONDecodeError:
                self._send_json({"error": "invalid request body"}, status=400)
                return True
            except FileManagerError as exc:
                self._send_json({"error": exc.message}, status=exc.status)
                return True
```

In do_GET, right after auth checks and before static file fallback, add:

```python
            if self._handle_file_manager_get(path, query):
                return
```

In do_POST, before final 404, add:

```python
            query = self._parse_query_params()
            if self._handle_file_manager_post(path, query, body):
                return
```

Add do_PUT:

```python
        def do_PUT(self) -> None:  # noqa: N802
            path = self._parse_request_path()
            query = self._parse_query_params()
            content_length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(content_length) if content_length > 0 else b""
            if self._handle_file_manager_put(path, query, body):
                return
            self._send_json({"error": "Not found"}, status=404)
```

---

### 7) Add File Manager UI Resources and Nav

File: orchestrator/web/static/index.html

In head, add:

```html
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/easymde/dist/easymde.min.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/vanilla-jsoneditor/themes/jse-theme-dark.css" />
  <link rel="stylesheet" href="/app-files.css" />
```

In desktop nav, insert Files immediately after the Recordings item:

```html
      <a href="#/home" class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-700 text-sm" data-nav="home" title="Home" aria-label="Home">
        <span aria-hidden="true">🏠</span>
        <span class="hidden lg:inline">Home</span>
      </a>
      <a href="#/music" class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-700 text-sm" data-nav="music" title="Music" aria-label="Music">
        <span aria-hidden="true">🎵</span>
        <span class="hidden lg:inline">Music</span>
      </a>
      <a href="#/recordings" class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-700 text-sm" data-nav="recordings" title="Recordings" aria-label="Recordings">
        <span aria-hidden="true">🎙️</span>
        <span class="hidden lg:inline">Recordings</span>
      </a>
      <a href="#/files" class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-700 text-sm" data-nav="files" title="Files" aria-label="Files">
        <span aria-hidden="true">📁</span>
        <span class="hidden lg:inline">Files</span>
      </a>
```

In mobile dropdown, insert Files immediately after Recordings:

```html
      <a href="#/home" class="flex items-center gap-2 px-4 py-2.5 hover:bg-gray-700 text-sm" data-nav="home">🏠 Home</a>
      <a href="#/music" class="flex items-center gap-2 px-4 py-2.5 hover:bg-gray-700 text-sm" data-nav="music">🎵 Music</a>
      <a href="#/recordings" class="flex items-center gap-2 px-4 py-2.5 hover:bg-gray-700 text-sm" data-nav="recordings">🎙️ Recordings</a>
      <a href="#/files" class="flex items-center gap-2 px-4 py-2.5 hover:bg-gray-700 text-sm" data-nav="files">📁 Files</a>
```

Before app scripts add EasyMDE and app-files:

```html
<script src="https://cdn.jsdelivr.net/npm/easymde/dist/easymde.min.js"></script>
<script src="/app-files.js"></script>
```

---

### 8) New File Manager CSS

File: orchestrator/web/static/app-files.css

```css
.fm-layout {
  display: grid;
  grid-template-columns: 280px 320px minmax(420px, 1fr);
  gap: 12px;
  height: 100%;
  min-height: 0;
}

.fm-panel {
  border: 1px solid rgb(55 65 81);
  border-radius: 12px;
  background: rgba(17, 24, 39, 0.45);
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.fm-scroll {
  overflow: auto;
  min-height: 0;
}

.fm-tree-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 8px;
  cursor: pointer;
}

.fm-tree-row:hover {
  background: rgba(55, 65, 81, 0.6);
}

.fm-tree-row.active {
  background: rgba(37, 99, 235, 0.35);
  border: 1px solid rgba(59, 130, 246, 0.6);
}

.fm-file-row {
  padding: 8px 10px;
  border-bottom: 1px solid rgb(31 41 55);
  cursor: pointer;
}

.fm-file-row:hover {
  background: rgba(31, 41, 55, 0.7);
}

.fm-file-row.active {
  background: rgba(37, 99, 235, 0.3);
}

.fm-editor-wrap {
  height: 100%;
  min-height: 0;
}

.fm-textarea {
  width: 100%;
  height: 100%;
  min-height: 420px;
  resize: vertical;
  border-radius: 10px;
  border: 1px solid rgb(55 65 81);
  background: rgb(17 24 39);
  color: rgb(229 231 235);
  padding: 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 13px;
  line-height: 1.45;
}

.fm-preview img,
.fm-preview video,
.fm-preview audio {
  max-width: 100%;
  border-radius: 10px;
}

.fm-modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 60;
}

.fm-modal {
  width: min(480px, calc(100vw - 40px));
  border: 1px solid rgb(55 65 81);
  border-radius: 12px;
  background: rgb(17 24 39);
  padding: 14px;
}

@media (max-width: 1200px) {
  .fm-layout {
    grid-template-columns: 250px minmax(260px, 1fr);
    grid-template-rows: 1fr 1fr;
  }
  .fm-layout .fm-panel:nth-child(3) {
    grid-column: 1 / -1;
  }
}

@media (max-width: 780px) {
  .fm-layout {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
}
```

---

### 9) New File Manager Frontend Logic

File: orchestrator/web/static/app-files.js

```javascript
(function () {
  const FM_API = '/api/file-manager';
  const FM_DEBOUNCE_MS = 500;

  const JSON_EDITOR_MODULE_URL = 'https://cdn.jsdelivr.net/npm/vanilla-jsoneditor/standalone.js';

  function fmEsc(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmState() {
    if (!S.fileManager) {
      S.fileManager = {
        initialized: false,
        loading: false,
        error: '',
        treeByPath: {},
        expandedByPath: { '/': true },
        selectedFolderPath: '/',
        selectedFilePath: '',
        folderChildren: [],
        currentFile: null,
        saveTimersByPath: {},
        saveStateByPath: {},
        activePlainEditorPath: '',
        activeMarkdownEditorPath: '',
        activeJsonEditorPath: '',
        markdownEditor: null,
        jsonEditor: null,
        createFolderModalOpen: false,
        createFolderName: '',
      };
    }
    return S.fileManager;
  }

  async function fmFetchJson(url, opts) {
    const response = await fetch(url, Object.assign({
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
    }, opts || {}));

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = String(payload && payload.error ? payload.error : 'request failed');
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  async function loadTree(path) {
    const st = fmState();
    const data = await fmFetchJson(FM_API + '/tree?path=' + encodeURIComponent(path));
    st.treeByPath[path] = Array.isArray(data.children) ? data.children : [];
    return st.treeByPath[path];
  }

  async function loadFolder(path) {
    const st = fmState();
    const data = await fmFetchJson(FM_API + '/folder?path=' + encodeURIComponent(path));
    st.folderChildren = Array.isArray(data.children) ? data.children : [];
    st.selectedFolderPath = path;
    return st.folderChildren;
  }

  async function loadFile(path) {
    const st = fmState();
    const data = await fmFetchJson(FM_API + '/file?path=' + encodeURIComponent(path));
    st.currentFile = data;
    st.selectedFilePath = path;
    st.activePlainEditorPath = '';
    st.activeMarkdownEditorPath = '';
    st.activeJsonEditorPath = '';
    return data;
  }

  async function saveFile(path, content) {
    const st = fmState();
    if (!st.currentFile || st.currentFile.path !== path) {
      return;
    }
    st.saveStateByPath[path] = 'saving';
    renderFileManagerPage(document.getElementById('main'));

    try {
      const res = await fmFetchJson(FM_API + '/file?path=' + encodeURIComponent(path), {
        method: 'PUT',
        body: JSON.stringify({
          content: String(content || ''),
          expectedEtag: String(st.currentFile.etag || ''),
        }),
      });
      st.currentFile.etag = res.etag;
      st.currentFile.size = res.size;
      st.currentFile.mtime = res.mtime;
      st.currentFile.content = String(content || '');
      st.saveStateByPath[path] = 'saved';
    } catch (err) {
      st.saveStateByPath[path] = String(err && err.status === 409 ? 'conflict' : 'error');
      st.error = String(err && err.message ? err.message : err);
    }

    renderFileManagerPage(document.getElementById('main'));
  }

  function queueSave(path, content) {
    const st = fmState();
    if (!path) return;
    if (st.saveTimersByPath[path]) {
      clearTimeout(st.saveTimersByPath[path]);
    }
    st.saveStateByPath[path] = 'dirty';
    st.saveTimersByPath[path] = setTimeout(() => {
      delete st.saveTimersByPath[path];
      void saveFile(path, content);
    }, FM_DEBOUNCE_MS);
    renderFileManagerPage(document.getElementById('main'));
  }

  function renderTreeRows(path, depth) {
    const st = fmState();
    const nodes = st.treeByPath[path] || [];
    return nodes.map((node) => {
      const nodePath = String(node.path || '');
      const isFolder = String(node.kind || '') === 'folder' || String(node.kind || '') === 'virtual-folder';
      const isExpanded = !!st.expandedByPath[nodePath];
      const isActive = st.selectedFolderPath === nodePath;
      const left = 10 + (depth * 14);

      const branch = isFolder
        ? '<button type="button" class="text-xs text-gray-300" data-action="fm-toggle-folder" data-path="' + fmEsc(nodePath) + '">' + (isExpanded ? '-' : '+') + '</button>'
        : '<span class="text-xs text-gray-500">.</span>';

      const row = ''
        + '<div class="fm-tree-row ' + (isActive ? 'active' : '') + '" style="margin-left:' + left + 'px" data-action="fm-select-folder" data-path="' + fmEsc(nodePath) + '">'
        + branch
        + '<span class="text-sm">' + fmEsc(node.name) + '</span>'
        + '</div>';

      if (isFolder && isExpanded) {
        return row + renderTreeRows(nodePath, depth + 1);
      }
      return row;
    }).join('');
  }

  function renderFolderRows() {
    const st = fmState();
    const items = Array.isArray(st.folderChildren) ? st.folderChildren : [];
    if (!items.length) {
      return '<div class="px-3 py-3 text-sm text-gray-400">Folder is empty</div>';
    }
    return items.map((item) => {
      const p = String(item.path || '');
      const kind = String(item.kind || 'file');
      const active = st.selectedFilePath === p;
      const icon = kind === 'folder' || kind === 'virtual-folder' ? 'DIR' : 'FILE';
      const action = kind === 'folder' || kind === 'virtual-folder' ? 'fm-select-folder' : 'fm-select-file';
      return ''
        + '<div class="fm-file-row ' + (active ? 'active' : '') + '" data-action="' + action + '" data-path="' + fmEsc(p) + '">'
        + '<div class="text-xs text-gray-500">' + icon + '</div>'
        + '<div class="text-sm truncate">' + fmEsc(item.name) + '</div>'
        + '</div>';
    }).join('');
  }

  function saveBadge() {
    const st = fmState();
    const file = st.currentFile;
    if (!file) return '';
    const value = String(st.saveStateByPath[file.path] || '');
    if (!value) return '';
    const label = value === 'dirty' ? 'Unsaved'
      : value === 'saving' ? 'Saving'
      : value === 'saved' ? 'Saved'
      : value === 'conflict' ? 'Conflict'
      : 'Error';
    return '<span class="px-2 py-1 rounded text-xs bg-gray-700">' + label + '</span>';
  }

  function renderEditorPane() {
    const st = fmState();
    const file = st.currentFile;
    if (!file) {
      return '<div class="px-4 py-4 text-sm text-gray-400">Select a file to open an editor/preview.</div>';
    }

    const category = String(file.category || 'binary');
    const editable = !!file.editable;
    const readOnly = editable ? '' : ('<div class="text-xs text-amber-300">' + fmEsc(file.readOnlyReason || 'Read-only file') + '</div>');

    if (category === 'media') {
      const mime = String(file.mimeType || '');
      const src = String(file.previewUrl || '');
      if (mime.startsWith('image/')) {
        return '<div class="fm-preview p-3"><img src="' + fmEsc(src) + '" alt="preview" /></div>';
      }
      if (mime.startsWith('audio/')) {
        return '<div class="fm-preview p-3"><audio controls src="' + fmEsc(src) + '" style="width:100%"></audio></div>';
      }
      if (mime.startsWith('video/')) {
        return '<div class="fm-preview p-3"><video controls src="' + fmEsc(src) + '" style="width:100%"></video></div>';
      }
      return '<div class="fm-preview p-3"><a class="underline" href="' + fmEsc(src) + '" target="_blank" rel="noreferrer noopener">Open media preview</a></div>';
    }

    if (category === 'binary') {
      return ''
        + '<div class="p-4 space-y-2">'
        + '<div class="text-sm text-gray-300">Binary file preview only.</div>'
        + '<a class="underline" href="' + fmEsc(file.previewUrl || '') + '" target="_blank" rel="noreferrer noopener">Open file</a>'
        + '</div>';
    }

    if (category === 'markdown') {
      return ''
        + '<div class="p-3 space-y-2">'
        + readOnly
        + '<textarea id="fmMarkdownEditor" class="fm-textarea">' + fmEsc(file.content || '') + '</textarea>'
        + '</div>';
    }

    if (category === 'json') {
      return ''
        + '<div class="p-3 space-y-2">'
        + readOnly
        + '<div id="fmJsonEditor" class="fm-editor-wrap"></div>'
        + '</div>';
    }

    return ''
      + '<div class="p-3 space-y-2">'
      + readOnly
      + '<textarea id="fmTextEditor" class="fm-textarea">' + fmEsc(file.content || '') + '</textarea>'
      + '</div>';
  }

  function destroyEditors() {
    const st = fmState();
    if (st.markdownEditor && typeof st.markdownEditor.toTextArea === 'function') {
      st.markdownEditor.toTextArea();
      st.markdownEditor = null;
    }
    if (st.jsonEditor && typeof st.jsonEditor.destroy === 'function') {
      st.jsonEditor.destroy();
      st.jsonEditor = null;
    }
  }

  async function mountEditors() {
    const st = fmState();
    const file = st.currentFile;
    if (!file) return;

    const category = String(file.category || 'binary');
    if (!file.editable) return;

    if (category === 'text') {
      const text = document.getElementById('fmTextEditor');
      if (text && st.activePlainEditorPath !== file.path) {
        st.activePlainEditorPath = file.path;
        text.addEventListener('input', () => queueSave(file.path, text.value));
      }
      return;
    }

    if (category === 'markdown') {
      const textarea = document.getElementById('fmMarkdownEditor');
      if (!textarea || st.activeMarkdownEditorPath === file.path) return;
      if (typeof EasyMDE !== 'function') {
        st.error = 'EasyMDE is not loaded';
        return;
      }
      st.activeMarkdownEditorPath = file.path;
      st.markdownEditor = new EasyMDE({
        element: textarea,
        autofocus: false,
        spellChecker: false,
        forceSync: true,
        status: false,
      });
      st.markdownEditor.value(String(file.content || ''));
      st.markdownEditor.codemirror.on('change', () => {
        queueSave(file.path, st.markdownEditor.value());
      });
      return;
    }

    if (category === 'json') {
      const mount = document.getElementById('fmJsonEditor');
      if (!mount || st.activeJsonEditorPath === file.path) return;

      st.activeJsonEditorPath = file.path;
      let createJSONEditor = null;
      try {
        const mod = await import(JSON_EDITOR_MODULE_URL);
        createJSONEditor = mod.createJSONEditor;
      } catch (err) {
        st.error = 'Failed to load vanilla-jsoneditor: ' + String(err && err.message ? err.message : err);
        renderFileManagerPage(document.getElementById('main'));
        return;
      }

      let parsed = {};
      try {
        parsed = JSON.parse(String(file.content || '{}'));
      } catch (_) {
        parsed = {};
      }

      st.jsonEditor = createJSONEditor({
        target: mount,
        props: {
          content: { json: parsed },
          mode: 'tree',
          onChange: (updatedContent) => {
            let nextText = '{}';
            if (updatedContent && updatedContent.text !== undefined) {
              nextText = String(updatedContent.text || '{}');
            } else if (updatedContent && updatedContent.json !== undefined) {
              nextText = JSON.stringify(updatedContent.json, null, 2);
            }
            queueSave(file.path, nextText);
          },
        },
      });
    }
  }

  async function ensureInitialized() {
    const st = fmState();
    if (st.initialized) return;
    st.loading = true;
    st.error = '';
    renderFileManagerPage(document.getElementById('main'));
    try {
      await loadTree('/');
      await loadFolder('/');
      st.initialized = true;
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
    } finally {
      st.loading = false;
      renderFileManagerPage(document.getElementById('main'));
      void mountEditors();
    }
  }

  async function selectFolder(path) {
    const st = fmState();
    st.error = '';
    st.selectedFolderPath = path;
    st.selectedFilePath = '';
    st.currentFile = null;
    destroyEditors();
    renderFileManagerPage(document.getElementById('main'));
    try {
      await loadFolder(path);
      if (!st.treeByPath[path]) {
        await loadTree(path);
      }
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
    }
    renderFileManagerPage(document.getElementById('main'));
  }

  async function selectFile(path) {
    const st = fmState();
    st.error = '';
    st.selectedFilePath = path;
    destroyEditors();
    renderFileManagerPage(document.getElementById('main'));
    try {
      await loadFile(path);
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
    }
    renderFileManagerPage(document.getElementById('main'));
    void mountEditors();
  }

  async function toggleTreeFolder(path) {
    const st = fmState();
    st.expandedByPath[path] = !st.expandedByPath[path];
    if (st.expandedByPath[path] && !st.treeByPath[path]) {
      try {
        await loadTree(path);
      } catch (err) {
        st.error = String(err && err.message ? err.message : err);
      }
    }
    renderFileManagerPage(document.getElementById('main'));
  }

  async function createFolder() {
    const st = fmState();
    const name = String(st.createFolderName || '').trim();
    if (!name) {
      st.error = 'Folder name is required';
      renderFileManagerPage(document.getElementById('main'));
      return;
    }

    try {
      await fmFetchJson(FM_API + '/folder?path=' + encodeURIComponent(st.selectedFolderPath || '/'), {
        method: 'POST',
        body: JSON.stringify({ name }),
      });
      st.createFolderModalOpen = false;
      st.createFolderName = '';
      delete st.treeByPath[st.selectedFolderPath || '/'];
      await selectFolder(st.selectedFolderPath || '/');
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
      renderFileManagerPage(document.getElementById('main'));
    }
  }

  window.renderFileManagerPage = function renderFileManagerPage(main) {
    const st = fmState();
    if (!main) return;
    main.dataset.page = 'files';

    const treeRows = renderTreeRows('/', 0);
    const folderRows = renderFolderRows();
    const editor = renderEditorPane();

    const modal = st.createFolderModalOpen
      ? ''
        + '<div class="fm-modal-backdrop">'
        + '<div class="fm-modal space-y-3">'
        + '<div class="text-sm font-semibold">Create folder</div>'
        + '<input id="fmCreateFolderName" type="text" value="' + fmEsc(st.createFolderName || '') + '" class="w-full rounded bg-gray-800 border border-gray-700 px-3 py-2 text-sm" placeholder="Folder name" />'
        + '<div class="flex justify-end gap-2">'
        + '<button type="button" class="px-3 py-1.5 rounded bg-gray-700" data-action="fm-create-cancel">Cancel</button>'
        + '<button type="button" class="px-3 py-1.5 rounded bg-blue-700" data-action="fm-create-confirm">Create</button>'
        + '</div>'
        + '</div>'
        + '</div>'
      : '';

    if (st.loading) {
      main.innerHTML = '<div class="px-4 py-4 text-sm text-gray-400">Loading file manager...</div>';
      return;
    }

    main.innerHTML = ''
      + '<div class="h-full min-h-0 p-2">'
      + '<div class="fm-layout">'
      + '<section class="fm-panel">'
      + '<div class="px-3 py-2 border-b border-gray-800 text-sm font-semibold">Workspace Tree</div>'
      + '<div class="fm-scroll px-2 py-2">' + treeRows + '</div>'
      + '</section>'
      + '<section class="fm-panel">'
      + '<div class="px-3 py-2 border-b border-gray-800 flex items-center justify-between gap-2">'
      + '<div class="text-sm font-semibold truncate">Folder Contents</div>'
      + '<button type="button" class="px-2 py-1 text-xs rounded bg-blue-700 hover:bg-blue-600" data-action="fm-open-create-folder">Create Folder</button>'
      + '</div>'
      + '<div class="fm-scroll">' + folderRows + '</div>'
      + '</section>'
      + '<section class="fm-panel">'
      + '<div class="px-3 py-2 border-b border-gray-800 flex items-center justify-between gap-2">'
      + '<div class="text-sm font-semibold truncate">'
      + fmEsc(st.currentFile ? st.currentFile.name : 'Editor / Preview')
      + '</div>'
      + saveBadge()
      + '</div>'
      + '<div class="fm-scroll">' + editor + '</div>'
      + '</section>'
      + '</div>'
      + (st.error ? '<div class="px-2 pt-2 text-xs text-red-300">' + fmEsc(st.error) + '</div>' : '')
      + '</div>'
      + modal;

    if (st.createFolderModalOpen) {
      setTimeout(() => {
        const input = document.getElementById('fmCreateFolderName');
        if (input) input.focus();
      }, 0);
    }

    setTimeout(() => {
      void mountEditors();
    }, 0);
  };

  window.handleFileManagerClick = function handleFileManagerClick(target, event) {
    const st = fmState();
    const toggle = target.closest('[data-action="fm-toggle-folder"]');
    if (toggle) {
      event.preventDefault();
      void toggleTreeFolder(String(toggle.dataset.path || '/'));
      return true;
    }

    const selectFolderBtn = target.closest('[data-action="fm-select-folder"]');
    if (selectFolderBtn) {
      event.preventDefault();
      const path = String(selectFolderBtn.dataset.path || '/');
      void selectFolder(path);
      return true;
    }

    const selectFileBtn = target.closest('[data-action="fm-select-file"]');
    if (selectFileBtn) {
      event.preventDefault();
      const path = String(selectFileBtn.dataset.path || '');
      if (path) void selectFile(path);
      return true;
    }

    const openCreate = target.closest('[data-action="fm-open-create-folder"]');
    if (openCreate) {
      event.preventDefault();
      st.createFolderModalOpen = true;
      st.createFolderName = '';
      renderFileManagerPage(document.getElementById('main'));
      return true;
    }

    const cancelCreate = target.closest('[data-action="fm-create-cancel"]');
    if (cancelCreate) {
      event.preventDefault();
      st.createFolderModalOpen = false;
      st.createFolderName = '';
      renderFileManagerPage(document.getElementById('main'));
      return true;
    }

    const confirmCreate = target.closest('[data-action="fm-create-confirm"]');
    if (confirmCreate) {
      event.preventDefault();
      void createFolder();
      return true;
    }

    return false;
  };

  window.handleFileManagerInput = function handleFileManagerInput(target) {
    const st = fmState();
    if (!target) return false;
    if (target.id === 'fmCreateFolderName') {
      st.createFolderName = String(target.value || '');
      return true;
    }
    return false;
  };

  window.handleFileManagerKeydown = function handleFileManagerKeydown(event) {
    const st = fmState();
    const t = event.target;
    if (!t) return false;
    if (t.id === 'fmCreateFolderName' && event.key === 'Enter') {
      event.preventDefault();
      void createFolder();
      return true;
    }
    if (event.key === 'Escape' && st.createFolderModalOpen) {
      st.createFolderModalOpen = false;
      st.createFolderName = '';
      renderFileManagerPage(document.getElementById('main'));
      return true;
    }
    return false;
  };

  window.ensureFileManagerReady = function ensureFileManagerReady() {
    void ensureInitialized();
  };
})();
```

---

### 10) Page Routing Updates

File: orchestrator/web/static/app-core.js

Update getPage:

```javascript
function getPage(){
    const h=location.hash.replace('#','');
    if(h==='/music') return 'music';
    if(h==='/recordings') return 'recordings';
    if(h==='/files') return 'files';
    return 'home';
}
```

Update getScrollUpArea:

```javascript
function getScrollUpArea(){
    if(S.page==='home') return document.getElementById('chatArea');
    if(S.page==='music') return document.getElementById('main');
    if(S.page==='recordings') return document.getElementById('main');
    if(S.page==='files') return document.getElementById('main');
    return null;
}
```

---

### 11) Event Delegation and Render Routing

File: orchestrator/web/static/app-events.js

Near top of click handler add:

```javascript
    if (S.page === 'files' && typeof handleFileManagerClick === 'function') {
        if (handleFileManagerClick(target, e)) {
            return;
        }
    }
```

In keydown listener add:

```javascript
    if (S.page === 'files' && typeof handleFileManagerKeydown === 'function') {
        if (handleFileManagerKeydown(e)) return;
    }
```

In input/search handlers add:

```javascript
document.addEventListener('input', e => {
    if (S.page === 'files' && typeof handleFileManagerInput === 'function') {
        if (handleFileManagerInput(e.target)) return;
    }
    handleTextInputChange(e.target);
});

document.addEventListener('search', e => {
    if (S.page === 'files' && typeof handleFileManagerInput === 'function') {
        if (handleFileManagerInput(e.target)) return;
    }
    handleTextInputChange(e.target);
});
```

In renderPage add files branch:

```javascript
    if(S.page==='music'){
        renderMusicPage(main);
        sendAction({type:'music_list_playlists'});
    } else if(S.page==='recordings'){
        renderRecordingsPage(main);
        if(!Array.isArray(S.recordings) || S.recordings.length===0){
            sendAction({type:'recordings_list'});
        }
    } else if (S.page==='files') {
        renderFileManagerPage(main);
        if (typeof ensureFileManagerReady === 'function') {
            ensureFileManagerReady();
        }
    } else {
        renderHomePage(main);
    }
```

---

### 12) WebSocket Navigation Update

File: orchestrator/web/static/app-ws.js

Update navigate handling:

```javascript
        case 'navigate':
            if(msg.page==='music' || msg.page==='home' || msg.page==='recordings' || msg.page==='files'){
                navigate(msg.page);
            }
            break;
```

Add filesystem-change handling for immediate live updates:

```javascript
        case 'file_manager_fs_changed':
            if(typeof handleFileManagerFsChanged === 'function'){
                handleFileManagerFsChanged(msg);
            }
            break;
```

### 13) Frontend Live-Update Hook

File: orchestrator/web/static/app-files.js

Add handler to consume websocket push events and keep UI synchronized immediately:

```javascript
  window.handleFileManagerFsChanged = function handleFileManagerFsChanged(msg) {
    const st = fmState();
    const resync = !!(msg && msg.resyncRequired);
    const changedPaths = Array.isArray(msg && msg.paths) ? msg.paths : [];

    if (resync) {
      st.treeByPath = {};
      st.folderChildren = [];
      if (S.page === 'files') {
        void ensureInitialized();
      }
      return;
    }

    for (const key of Object.keys(st.treeByPath || {})) {
      if (key === '/' || changedPaths.some((p) => String(p || '').startsWith(String(key || '') + '/'))) {
        delete st.treeByPath[key];
      }
    }

    if (S.page === 'files') {
      const folder = String(st.selectedFolderPath || '/');
      void loadTree('/').then(() => loadFolder(folder)).then(() => {
        if (st.selectedFilePath) {
          return loadFile(st.selectedFilePath).catch(() => null);
        }
        return null;
      }).finally(() => {
        renderFileManagerPage(document.getElementById('main'));
      });
    }
  };
```

---

## Test Plan

### Backend
- path traversal blocked
- excluded folders not returned in tree/folder APIs
- top-level config files hidden from root and visible in virtual folder
- media preview endpoint serves allowed files
- etag conflict returns 409
- create folder validates name and rejects excluded names
- inotify overflow and strict-limit branches emit websocket resyncRequired=true
- watcher honors max_watches/max_events_per_tick/max_paths_per_push limits

### Frontend
- tree expands lazily and remembers expanded nodes
- selecting folder updates file list
- selecting file mounts correct editor/preview
- markdown/json/text autosave after debounce
- conflict state shown when stale save occurs
- create-folder modal creates folder and refreshes list/tree
- websocket file_manager_fs_changed event updates current page immediately without manual refresh

### Manual
- verify excluded folder list from config
- verify OpenClaw Configuration synthetic node at root
- verify markdown editor toolbar and keyboard shortcuts
- verify JSON tree expand/edit/save
- verify media preview for image/audio/video
- modify files/folders from shell while Files page is open and verify instant UI updates
- verify Files menu appears after Recordings on desktop and mobile menus

## Delivery Notes
This file now contains the implementation-ready code pack and resolved editor recommendations. No additional planning decisions remain open for v1.
