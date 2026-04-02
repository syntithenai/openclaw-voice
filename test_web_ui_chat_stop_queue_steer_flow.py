from pathlib import Path


def test_web_ui_chat_stop_queue_steer_flow_contract() -> None:
    html = Path("orchestrator/web/static/index.html").read_text(encoding="utf-8")
    events = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")
    ws = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    assert 'id="chatModeQueueBtn"' in html
    assert 'id="chatModeSteerBtn"' in html
    assert 'id="chatStopBtn"' in html
    assert 'id="chatQueueList"' in html
    assert "function requestChatStop()" in events
    assert "function processQueuedChatDispatch()" in events
    assert "type:'chat_steer_now'" in events
    assert "case 'chat_stop_ack':" in ws
    assert "case 'chat_steer_ack':" in ws
    assert "case 'chat_run_state':" in ws
