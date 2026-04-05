from pathlib import Path


def test_chat_stream_gap_reconcile_and_partial_stream_contract() -> None:
    ws_source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    assert "function acceptChatFrame(msg){" in ws_source
    assert "if(next.seq > 0 && seq > (next.seq + 1)){" in ws_source
    assert "sendAction({type:'chat_request_reconcile', request_id:rid, last_seq:next.seq});" in ws_source
    assert "scheduleAssistantStreamBubbleUpdate('active', nextMsg);" in ws_source
    assert "isAssistantStreamBubbleActive()" in ws_source