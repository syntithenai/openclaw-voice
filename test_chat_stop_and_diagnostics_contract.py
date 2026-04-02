from pathlib import Path


def test_realtime_service_handles_chat_stop_and_diagnostics_actions() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert 'if msg_type == "chat_stop" and self._on_chat_stop:' in source
    assert 'await _send_ws_json({"type": "chat_stop_ack", "ok": True})' in source
    assert 'await _send_ws_json({"type": "chat_stop_error", "ok": False, "error": str(exc)})' in source

    assert 'if msg_type == "chat_verbose_set" and self._on_chat_verbose_set:' in source
    assert 'if msg_type == "chat_reasoning_set" and self._on_chat_reasoning_set:' in source
    assert 'if msg_type == "chat_lifecycle_policy_set" and self._on_chat_lifecycle_policy_set:' in source
    assert 'if msg_type == "chat_interim_set" and self._on_chat_interim_set:' in source

    assert '"type": "setting_action_ack"' in source
    assert '"type": "setting_action_error"' in source
    assert 'if msg_type == "chat_stream_ack":' in source
    assert 'if msg_type == "chat_request_reconcile":' in source
    assert '"type": "chat_reconcile_snapshot"' in source
    assert 'if msg_type == "chat_stream_replay":' in source
    assert 'if msg_type == "chat_steer_now" and self._on_chat_steer_now:' in source
    assert '"type": "chat_steer_ack"' in source
    assert '"type": "chat_steer_error"' in source


def test_chat_ws_triggers_queue_dispatch_and_stop_state_transitions() -> None:
    source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    # Queue drain should be attempted whenever key chat update events land.
    assert "if(typeof processQueuedChatDispatch==='function') processQueuedChatDispatch();" in source
    assert "case 'chat_text_ack':" in source
    assert "case 'chat_append':" in source
    assert "case 'chat_update':" in source

    assert "case 'chat_stop_ack':" in source
    assert "S.chatQueuedItems = [];" in source
    assert "case 'chat_stop_error':" in source
    assert "case 'chat_queue_update':" in source
    assert "case 'chat_steer_ack':" in source
    assert "case 'chat_steer_error':" in source
    assert "recordInlineError('setting', 'chat_stop', String(msg.error||'Failed to stop chat'));" in source


def test_orchestrator_applies_chat_debug_visibility_policy_guards() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "def _chat_detail_enabled() -> bool:" in source
    assert "def _debug_log_enabled() -> bool:" in source

    # Chat timeline and debug streams must be independently gateable.
    assert "if not hasattr(gateway, \"listen_raw\") and _debug_log_enabled():" in source
    assert "if step_msg and _chat_detail_enabled():" in source
    assert "if interim_msg and _chat_detail_enabled() and web_chat_interim_enabled:" in source
    assert "if not _debug_log_enabled():" in source
    assert "if _debug_log_enabled():" in source

    # Verbose level drives whether event details are hidden/short/full.
    assert "if web_chat_verbose_level == \"full\":" in source
    assert "elif web_chat_verbose_level == \"on\":" in source


def test_orchestrator_registers_new_chat_setting_handlers() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "on_chat_stop=_ui_chat_stop," in source
    assert "on_chat_verbose_set=_ui_chat_verbose_set," in source
    assert "on_chat_reasoning_set=_ui_chat_reasoning_set," in source
    assert "on_chat_lifecycle_policy_set=_ui_chat_lifecycle_policy_set," in source
    assert "on_chat_interim_set=_ui_chat_interim_set," in source


def test_chat_ui_exposes_diagnostics_controls_and_pref_push() -> None:
    html_source = Path("orchestrator/web/static/index.html").read_text(encoding="utf-8")
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert 'id="chatVerboseBtn"' in html_source
    assert 'id="chatReasoningBtn"' in html_source
    assert 'id="chatLifecyclePolicyBtn"' in html_source
    assert 'id="chatInterimToggle"' in html_source
    assert 'id="chatRunElapsed"' in html_source

    assert "sendSettingValueAction('chat_verbose_set'" in core_source
    assert "sendSettingValueAction('chat_reasoning_set'" in core_source
    assert "sendSettingValueAction('chat_lifecycle_policy_set'" in core_source
    assert "sendSettingAction('chat_interim_set'" in core_source


def test_orchestrator_includes_generation_and_stream_sequence_fencing() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "def _active_gateway_collation_context() -> tuple[int, int]:" in source
    assert "def _next_stream_seq(request_id: int) -> int:" in source
    assert "def _register_terminal_winner(request_id: int, generation: int, classification: str) -> bool:" in source
    assert "def _frame_generation(payload: dict[str, Any], request_id: int, fallback_generation: int) -> int:" in source

    assert '"request_generation": int(request_generation)' in source
    assert '"stream_seq": _next_stream_seq(request_id)' in source
    assert '"event_ts": time.time(),' in source
    assert "if payload_generation < request_generation:" in source
    assert '"type": "chat_run_state",' in source
    assert "def _emit_chat_run_state(" in source


def test_chat_client_drops_stale_or_out_of_order_frames() -> None:
    source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    assert "function acceptChatFrame(msg){" in source
    assert "if(next.generation > 0 && generation < next.generation) return false;" in source
    assert "if(seq <= next.seq) return false;" in source
    assert "sendAction({type:'chat_request_reconcile', request_id:rid, last_seq:next.seq});" in source
    assert "case 'chat_run_state':" in source
    assert "case 'chat_reconcile_snapshot':" in source
    assert "case 'chat_stream_replay':" in source
    assert "if(nextMsg && !acceptChatFrame(nextMsg)) break;" in source
    assert "if(updatedMsg && !acceptChatFrame(updatedMsg)) break;" in source
