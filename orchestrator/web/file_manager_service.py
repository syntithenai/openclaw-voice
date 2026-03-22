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
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
        ".scss",
        ".html",
        ".htm",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".sh",
        ".bash",
        ".zsh",
        ".md",
        ".sql",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".log",
        ".csv",
    }
    MEDIA_EXTS = {
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".mp4",
        ".mkv",
        ".mov",
        ".webm",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
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

    def _is_excluded_path(self, candidate: Path) -> bool:
        if candidate == self.root:
            return False
        try:
            rel_parts = candidate.relative_to(self.root).parts
        except ValueError:
            return True
        return any(part in self.excluded_folders for part in rel_parts)

    def _resolve_real(self, api_path: str) -> Path:
        normalized = self._normalize_api_path(api_path)
        rel = normalized.lstrip("/")
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise FileManagerError(400, "path escapes workspace root")
        if self._is_excluded_path(candidate):
            raise FileManagerError(404, "path is excluded")
        return candidate

    def _resolve_virtual_file(self, api_path: str) -> Path:
        normalized = self._normalize_api_path(api_path)
        if not normalized.startswith(self.VIRTUAL_CONFIG_ROOT + "/"):
            raise FileManagerError(400, "invalid virtual file path")
        suffix = normalized[len(self.VIRTUAL_CONFIG_ROOT + "/") :]
        if not suffix or "/" in suffix:
            raise FileManagerError(400, "invalid virtual file path")
        file_name = suffix
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
