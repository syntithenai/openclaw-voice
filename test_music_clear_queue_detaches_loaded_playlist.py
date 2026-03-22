import asyncio
import random

import pytest

from orchestrator.music.manager import MusicManager
from orchestrator.music.parser import MusicFastPathParser
from orchestrator.music.router import MusicRouter


class FakePool:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def execute(self, command: str, timeout=None):
        self.commands.append(command)
        return {}


def test_clear_queue_detaches_loaded_saved_playlist() -> None:
    pool = FakePool()
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)
    manager._loaded_playlist_name = "Roadtrip Mix"

    result = asyncio.run(manager.clear_queue())

    assert result == "Queue cleared"
    assert pool.commands == ["clear"]
    assert manager.get_loaded_playlist_name() == ""


def test_clear_queue_without_loaded_playlist_still_succeeds() -> None:
    pool = FakePool()
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)

    result = asyncio.run(manager.clear_queue())

    assert result == "Queue cleared"
    assert pool.commands == ["clear"]
    assert manager.get_loaded_playlist_name() == ""


def test_list_playlists_returns_backend_names() -> None:
    class PlaylistPool(FakePool):
        async def execute_list(self, command: str, timeout=None):
            assert command == "listplaylists"
            return [{"playlist": "Roadtrip"}, {"playlist": "Ambient"}]

    pool = PlaylistPool()
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)

    playlists = asyncio.run(manager.list_playlists())

    assert playlists == ["Roadtrip", "Ambient"]


def test_delete_active_playlist_clears_queue_and_detaches() -> None:
    class PlaylistPool(FakePool):
        async def execute_list(self, command: str, timeout=None):
            assert command == "listplaylists"
            return [{"playlist": "Roadtrip Mix"}, {"playlist": "Ambient"}]

    pool = PlaylistPool()
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)
    manager._loaded_playlist_name = "Roadtrip Mix"

    result = asyncio.run(manager.delete_playlist("roadtrip mix"))

    assert result == "Deleted playlist: roadtrip mix"
    assert pool.commands == ['rm "Roadtrip Mix"', "clear"]
    assert manager.get_loaded_playlist_name() == ""


def test_delete_inactive_playlist_keeps_existing_queue_association() -> None:
    class PlaylistPool(FakePool):
        async def execute_list(self, command: str, timeout=None):
            assert command == "listplaylists"
            return [{"playlist": "Roadtrip Mix"}, {"playlist": "Ambient"}]

    pool = PlaylistPool()
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)
    manager._loaded_playlist_name = "Roadtrip Mix"

    result = asyncio.run(manager.delete_playlist("Ambient"))

    assert result == "Deleted playlist: Ambient"
    assert pool.commands == ['rm "Ambient"']
    assert manager.get_loaded_playlist_name() == "Roadtrip Mix"


def test_parser_routes_clear_queue_phrases_to_clear_queue_command() -> None:
    parser = MusicFastPathParser()

    assert parser.parse("clear the queue") == ("clear_queue", {})
    assert parser.parse("empty queue") == ("clear_queue", {})


def test_music_router_music_clear_queue_tool_calls_manager_clear_queue() -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.clear_calls = 0

        async def clear_queue(self) -> str:
            self.clear_calls += 1
            return "Queue cleared"

    manager = FakeManager()
    router = MusicRouter(manager)

    result = asyncio.run(router.handle_tool_call("music_clear_queue", {}))

    assert result == "Queue cleared."
    assert manager.clear_calls == 1


def test_play_genre_samples_full_match_set_and_spreads_artists(monkeypatch: pytest.MonkeyPatch) -> None:
    tracks = [
        {"file": "a1.mp3", "Artist": "Artist A"},
        {"file": "a2.mp3", "Artist": "Artist A"},
        {"file": "a3.mp3", "Artist": "Artist A"},
        {"file": "b1.mp3", "Artist": "Artist B"},
        {"file": "b2.mp3", "Artist": "Artist B"},
        {"file": "c1.mp3", "Artist": "Artist C"},
    ]

    class GenrePool(FakePool):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list[list[str]] = []
            self.search_commands: list[str] = []

        async def execute(self, command: str, timeout=None):
            self.commands.append(command)
            if command == "status":
                return {"state": "play", "volume": "50"}
            return {}

        async def execute_list(self, command: str, timeout=None):
            if command.startswith('search genre "pop"'):
                self.search_commands.append(command)
                return list(tracks)
            if command == "outputs":
                return [{"outputname": "Speaker", "outputenabled": "1"}]
            raise AssertionError(f"Unexpected execute_list command: {command}")

        async def execute_batch(self, commands: list[str], timeout=None):
            self.batches.append(list(commands))
            return {}

    monkeypatch.setattr(random, "sample", lambda population, k: [population[0], population[1], population[3], population[5]])
    monkeypatch.setattr(random, "shuffle", lambda values: None)

    pool = GenrePool()
    manager = MusicManager(pool, genre_queue_limit=4, pipewire_stream_normalize_enabled=False)

    result = asyncio.run(manager.play_genre("pop", shuffle=True))

    assert result == "Playing 4 pop tracks"
    assert pool.search_commands == ['search genre "pop" window 0:999999']
    assert pool.batches == [[
        'add "a1.mp3"',
        'add "b1.mp3"',
        'add "c1.mp3"',
        'add "a2.mp3"',
    ]]
    assert pool.commands == ["clear", "play 0", "random 1", "status"]
