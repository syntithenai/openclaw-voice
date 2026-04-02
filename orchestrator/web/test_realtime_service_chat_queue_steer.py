from pathlib import Path


def test_chat_queue_steer_routes_exist() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert 'if msg_type == "chat_text" and self._on_chat_text:' in source
    assert 'if msg_type == "chat_steer_now" and self._on_chat_steer_now:' in source
    assert '"type": "chat_steer_ack"' in source
    assert '"type": "chat_steer_error"' in source
    assert 'if msg_type == "chat_stop" and self._on_chat_stop:' in source
