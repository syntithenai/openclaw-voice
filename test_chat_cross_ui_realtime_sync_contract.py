from pathlib import Path


def test_cross_ui_realtime_sync_loop_contract() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "async def _web_ui_chat_session_sync() -> None:" in source
    assert "sessions = await gateway.list_sessions(" in source
    assert "mapped_messages = await _ui_chat_load_thread_messages(active_thread_id, \"sync\")" in source
    assert "web_service.select_chat_thread(active_thread_id, messages_override=mapped_messages)" in source
    assert "asyncio.create_task(_web_ui_chat_session_sync())" in source


def test_cross_ui_thread_loader_handler_wired_contract() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "async def _ui_chat_load_thread_messages(client_thread_id: str, client_id: str)" in source
    assert "from orchestrator.gateway.session_mapper import map_gateway_messages_to_voice_format" in source
    assert "on_chat_load_thread_messages=_ui_chat_load_thread_messages," in source
