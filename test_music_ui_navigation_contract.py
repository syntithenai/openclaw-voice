from pathlib import Path

from orchestrator.gateway.quick_answer import classify_upstream_decision
from orchestrator.music.parser import MusicFastPathParser


def test_queue_queries_are_classified_as_music_related() -> None:
    parser = MusicFastPathParser()

    assert parser.is_music_related("change what is in the queue")
    assert parser.is_music_related("load the queued playlist")


def test_queue_queries_stay_on_local_music_path() -> None:
    should_use_upstream, reason = classify_upstream_decision(
        "change what is in the queue",
        music_enabled=True,
    )

    assert not should_use_upstream
    assert reason == "music_local"


def test_music_page_navigation_requires_explicit_user_navigation() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "def _should_navigate_to_music_page(" not in source
    assert "web_service.navigate_ui_page(" not in source