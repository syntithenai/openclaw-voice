from __future__ import annotations

import asyncio

import pytest

from orchestrator.music import native_backend as music_client


class _FakeProc:
    """Stand-in for asyncio.subprocess.Process. wait() returns immediately."""
    returncode: int | None = 0

    async def wait(self) -> int:
        return 0


class _FakeLibrary:
    def get_track(self, file_uri: str) -> dict:
        return {"file": file_uri, "duration": "120"}


def _run_loop_one_advance(backend, timeout: float = 1.0) -> None:
    """Run _auto_advance_loop until it has performed one advance, then cancel."""
    async def _runner():
        task = asyncio.create_task(backend._auto_advance_loop())
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    asyncio.run(_runner())


def test_loop_advances_to_next_track_when_process_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop must advance to position 1 when the proc at position 0 exits naturally."""
    backend = music_client._BACKEND
    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state

    played_files: list[str] = []
    fake_proc = _FakeProc()

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        played_files.append(file_uri)
        backend.player._proc = None  # prevent loop from spawning again within timeout
        return True

    try:
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=101),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=102),
        ]
        backend.current_pos = 0
        backend.state = "play"
        backend.player._proc = fake_proc
        backend.player.output_route = "local"

        monkeypatch.setattr(backend, "library", _FakeLibrary())
        monkeypatch.setattr(backend.player, "play", fake_play)

        _run_loop_one_advance(backend)

        assert played_files == ["Artist/Album/second.mp3"]
        assert backend.current_pos == 1
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state
        backend.player.output_route = "local"


def test_loop_wraps_from_last_track_to_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop must wrap from position 1 (last) back to position 0 and continue playing."""
    backend = music_client._BACKEND
    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state

    played_files: list[str] = []
    fake_proc = _FakeProc()

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        played_files.append(file_uri)
        backend.player._proc = None
        return True

    try:
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=201),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=202),
        ]
        backend.current_pos = 1
        backend.state = "play"
        backend.player._proc = fake_proc
        backend.player.output_route = "local"

        monkeypatch.setattr(backend, "library", _FakeLibrary())
        monkeypatch.setattr(backend.player, "play", fake_play)

        _run_loop_one_advance(backend)

        assert played_files == ["Artist/Album/first.mp3"]
        assert backend.current_pos == 0
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state
        backend.player.output_route = "local"


def test_loop_aborts_when_user_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """If state != 'play' when proc exits, loop must NOT start a new track."""
    backend = music_client._BACKEND
    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state

    played_files: list[str] = []
    fake_proc = _FakeProc()

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        played_files.append(file_uri)
        return True

    async def _runner():
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=301),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=302),
        ]
        backend.current_pos = 0
        backend.state = "play"
        backend.player._proc = fake_proc
        backend.player.output_route = "local"

        monkeypatch.setattr(backend, "library", _FakeLibrary())
        monkeypatch.setattr(backend.player, "play", fake_play)

        task = asyncio.create_task(backend._auto_advance_loop())
        # Let one iteration run (proc.wait() returns immediately), then stop state
        # before the advance can fire by setting state to "stop" immediately.
        backend.state = "stop"
        try:
            await asyncio.wait_for(task, timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(_runner())
        assert played_files == []
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state
        backend.player.output_route = "local"


def test_loop_handles_seekcur_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop must skip advance when proc was replaced by seekcur, then advance for the NEW proc."""
    backend = music_client._BACKEND
    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state

    played_files: list[str] = []
    old_proc = _FakeProc()
    new_proc = _FakeProc()

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        played_files.append(file_uri)
        backend.player._proc = None
        return True

    try:
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=401),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=402),
        ]
        backend.current_pos = 0
        backend.state = "play"
        # Simulate seekcur already having replaced the proc
        backend.player._proc = new_proc  # seekcur proc
        backend.player.output_route = "local"

        monkeypatch.setattr(backend, "library", _FakeLibrary())
        monkeypatch.setattr(backend.player, "play", fake_play)

        # Run loop starting with new_proc (the seekcur replacement)
        _run_loop_one_advance(backend)

        # Should have advanced using the seekcur proc, not the old one
        assert played_files == ["Artist/Album/second.mp3"]
        assert backend.current_pos == 1
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state
        backend.player.output_route = "local"


def test_status_browser_route_advances_when_elapsed_reaches_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Browser output has no process; status must auto-advance when elapsed >= duration."""
    backend = music_client._BACKEND
    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state
    original_route = backend.player.output_route
    original_elapsed_anchor_ts = backend.elapsed_anchor_ts
    original_elapsed_anchor_value = backend.elapsed_anchor_value

    played_files: list[str] = []

    class _BrowserLibrary:
        def get_track(self, file_uri: str) -> dict:
            return {"file": file_uri, "duration": "10"}

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        del seek_s
        played_files.append(file_uri)
        backend.player.browser_stream_path = "/tmp/current.m4a"
        return True

    try:
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=501),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=502),
        ]
        backend.current_pos = 0
        backend.state = "play"
        backend.player.output_route = "browser"
        backend.elapsed_anchor_value = 10.0
        backend.elapsed_anchor_ts = 0.0

        monkeypatch.setattr(backend, "library", _BrowserLibrary())
        monkeypatch.setattr(backend.player, "play", fake_play)

        status = asyncio.run(backend.execute("status"))

        assert played_files == ["Artist/Album/second.mp3"]
        assert backend.current_pos == 1
        assert status["song"] == "1"
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state
        backend.player.output_route = original_route
        backend.elapsed_anchor_ts = original_elapsed_anchor_ts
        backend.elapsed_anchor_value = original_elapsed_anchor_value
