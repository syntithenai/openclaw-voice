from pathlib import Path


def test_thread_selection_requests_server_history_load() -> None:
    source = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")

    assert "S.chatThreadLoadPendingId = String(tid);" in source
    assert "sendAction({type:'chat_select', thread_id: tid});" in source


def test_websocket_bootstrap_refreshes_selected_thread() -> None:
    source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    assert "function requestSelectedChatThreadLoad(force){" in source
    assert "if(!S.chatThreadBootstrapRequested){" in source
    assert "requestSelectedChatThreadLoad(true);" in source
    assert "requestSelectedChatThreadLoad(false);" in source