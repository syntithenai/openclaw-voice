from pathlib import Path


def test_chat_select_routes_to_server_activation() -> None:
    ws_source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")
    realtime_source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "sendAction({type:'chat_select', thread_id: tid});" in ws_source
    assert "if msg_type == \"chat_select\":" in realtime_source
    assert "self.select_chat_thread(thread_id, messages_override=loaded_messages)" in realtime_source


def test_chat_reset_uses_server_payload_and_selection_pending_guard() -> None:
    ws_source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    assert "case 'chat_reset':" in ws_source
    assert "applyServerChatState(" in ws_source
    assert "Array.isArray(msg.chat) ? msg.chat : []" in ws_source
    assert "if(pendingTid && activeTid && pendingTid===activeTid) S.chatThreadLoadPendingId='';" in ws_source


def test_historic_thread_load_applies_messages_override() -> None:
    realtime_source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "def select_chat_thread(self, thread_id: str, messages_override: list[dict[str, Any]] | None = None) -> bool:" in realtime_source
    assert "if messages_override is not None:" in realtime_source
    assert "selected[\"messages\"] = list(messages_override)" in realtime_source


def test_active_edits_update_active_thread_and_broadcast_threads() -> None:
    realtime_source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "def append_chat_message(self, message: dict[str, Any]) -> None:" in realtime_source
    assert "def upsert_chat_message(self, message: dict[str, Any]) -> None:" in realtime_source
    assert "self._upsert_active_chat_thread()" in realtime_source
    assert '"chat_threads": list(self._chat_threads)' in realtime_source


def test_loaded_historic_session_selection_is_preserved_on_server_updates() -> None:
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert "const selectedBefore = String(S.selectedChatId||'active').trim() || 'active';" in core_source
    assert "let nextSelected = selectedBefore;" in core_source
    assert "if(nextSelected!=='active' && !selectedExists){" in core_source


def test_backend_marks_loaded_thread_as_active_chat_thread() -> None:
    realtime_source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "self._active_chat_id = \"active\"" in realtime_source
    assert "self._active_chat_thread_id = tid" in realtime_source


def test_new_session_archives_previous_active_chat() -> None:
    realtime_source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert "def start_new_chat(self) -> None:" in realtime_source
    assert "self._archive_active_chat_if_needed()" in realtime_source
    assert "self._active_chat_thread_id = None" in realtime_source


def test_frontend_keeps_pending_thread_selected_during_load() -> None:
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert "const pendingTid = String(S.chatThreadLoadPendingId||'').trim();" in core_source
    assert "if(nextSelected==='active' && pendingExists){" in core_source
    assert "nextSelected = pendingTid;" in core_source


def test_chat_threads_are_sorted_recent_first_in_frontend_merge() -> None:
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert "function mergeCachedRawGatewayThreads(serverThreads, cachedThreads){" in core_source
    assert ".sort((a,b)=>Number(b.updated_ts||0)-Number(a.updated_ts||0));" in core_source


def test_chat_state_is_server_authoritative_without_local_cache() -> None:
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert "function persistChatCache(){" in core_source
    assert "localStorage.removeItem(getChatCacheKey());" in core_source
    assert "function hydrateChatCache(){" in core_source
    assert "Intentionally no-op: chat state is loaded from websocket state snapshots." in core_source


def test_session_list_does_not_render_current_chat_row() -> None:
    events_source = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")

    assert "if(String(t.id||'')==='active') return false;" in events_source


def test_follow_latest_button_is_not_rendered() -> None:
    events_source = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")

    assert 'id="chatFollowToggle"' not in events_source
    assert 'data-action="chat-follow-toggle"' not in events_source
    assert "function updateChatFollowToggleState()" not in events_source


def test_chat_delete_uses_modal_confirmation_then_dispatches_action() -> None:
    events_source = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")

    assert "const deleteThreadConfirmBtn = target.closest('[data-action=\"chat-delete-confirm\"]');" in events_source
    assert "sendAction({type:'chat_delete', thread_id: tid});" in events_source
