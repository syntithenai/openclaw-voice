from pathlib import Path


def test_add_selected_sets_scroll_to_top_flag() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "musicQueueScrollToTopOnNextQueueRender:false" in source
    assert "S.musicQueueScrollToTopOnNextQueueRender = true;" in source
    assert "sendMusicAction('music_add_files', {{files}});" in source


def test_queue_render_consumes_scroll_to_top_flag() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "if(!S.musicAddMode && S.musicQueueScrollToTopOnNextQueueRender){{" in source
    assert "main.scrollTo({{top:0, behavior:'smooth'}});" in source
    assert "S.musicQueueScrollToTopOnNextQueueRender = false;" in source


def test_music_add_error_clears_scroll_to_top_flag() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "if(String(msg.action||'')==='music_add_files') S.musicQueueScrollToTopOnNextQueueRender=false;" in source
