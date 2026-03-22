from __future__ import annotations

import asyncio

import pytest

from orchestrator.music import native_backend as music_client


def test_status_advances_to_next_track_when_local_player_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = music_client._BACKEND

    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state

    played_files: list[str] = []

    class FakeLibrary:
        def get_track(self, file_uri: str):
            return {"file": file_uri, "duration": "120"}

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        del seek_s
        played_files.append(file_uri)
        return True

    try:
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=101),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=102),
        ]
        backend.current_pos = 0
        backend.state = "play"

        monkeypatch.setattr(backend, "library", FakeLibrary())
        monkeypatch.setattr(backend.player, "is_active", lambda: False)
        monkeypatch.setattr(backend.player, "play", fake_play)

        status = asyncio.run(backend.execute("status"))

        assert played_files == ["Artist/Album/second.mp3"]
        assert backend.current_pos == 1
        assert status["song"] == "1"
        assert status["state"] == "play"
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state



def test_status_wraps_to_first_track_after_last_track_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = music_client._BACKEND

    original_queue = list(backend.queue)
    original_pos = backend.current_pos
    original_state = backend.state

    played_files: list[str] = []

    class FakeLibrary:
        def get_track(self, file_uri: str):
            return {"file": file_uri, "duration": "180"}

    async def fake_play(file_uri: str, seek_s: int = 0) -> bool:
        del seek_s
        played_files.append(file_uri)
        return True

    try:
        backend.queue = [
            music_client.QueueItem(file="Artist/Album/first.mp3", id=201),
            music_client.QueueItem(file="Artist/Album/second.mp3", id=202),
        ]
        backend.current_pos = 1
        backend.state = "play"

        monkeypatch.setattr(backend, "library", FakeLibrary())
        monkeypatch.setattr(backend.player, "is_active", lambda: False)
        monkeypatch.setattr(backend.player, "play", fake_play)

        status = asyncio.run(backend.execute("status"))

        assert played_files == ["Artist/Album/first.mp3"]
        assert backend.current_pos == 0
        assert status["song"] == "0"
        assert status["state"] == "play"
    finally:
        backend.queue = original_queue
        backend.current_pos = original_pos
        backend.state = original_state