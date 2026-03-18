import asyncio

from orchestrator.music.manager import MusicManager


class FakePool:
    def __init__(self, status_responses):
        self._status_responses = list(status_responses)
        self.commands = []

    async def execute(self, command: str):
        self.commands.append(command)
        if command == "status":
            if self._status_responses:
                return self._status_responses.pop(0)
            return {}
        if command.startswith("addid "):
            return {"Id": str(len([cmd for cmd in self.commands if cmd.startswith("addid ")]))}
        return {}


def test_add_files_to_queue_inserts_at_top_in_requested_order() -> None:
    pool = FakePool([{"state": "play"}])
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)

    result = asyncio.run(manager.add_files_to_queue(["first.mp3", "second.mp3", "third.mp3"]))

    assert result == "Added 3 track(s) to queue"
    assert pool.commands == [
        "status",
        'addid "third.mp3" 0',
        'addid "second.mp3" 0',
        'addid "first.mp3" 0',
        "play 0",
    ]


def test_add_files_to_queue_keeps_new_top_track_selected_when_stopped() -> None:
    pool = FakePool([
        {"state": "stop"},
        {"state": "stop"},
    ])
    manager = MusicManager(pool, pipewire_stream_normalize_enabled=False)

    result = asyncio.run(manager.add_files_to_queue(["alpha.mp3", "beta.mp3"]))

    assert result == "Added 2 track(s) to queue"
    assert pool.commands == [
        "status",
        'addid "beta.mp3" 0',
        'addid "alpha.mp3" 0',
        "play 0",
        "stop",
        "status",
        "play 0",
        "pause 1",
    ]