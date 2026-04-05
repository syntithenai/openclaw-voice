from pathlib import Path


def test_chat_send_split_dropdown_opens_above_bar() -> None:
    html_source = Path("orchestrator/web/static/index.html").read_text(encoding="utf-8")

    assert 'id="chatSendDropdown"' in html_source
    assert 'style="bottom:calc(100% + 0.25rem)"' in html_source


def test_chat_mode_dropdown_buttons_submit_and_persist_mode() -> None:
    events_source = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")

    # Mode choice should submit immediately and persist mode via setChatSendMode.
    assert "function submitChatComposer(modeOverride){" in events_source
    assert "if (modeOverride) setChatSendMode(modeOverride);" in events_source
    assert "submitChatComposer('queue');" in events_source
    assert "submitChatComposer('steer');" in events_source


def test_chat_send_controls_disabled_when_input_empty() -> None:
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert "const hasInputText = !!(input && String(input.value||'').trim());" in core_source
    assert "const sendDisabled = isPending || S.chatStopPending || !hasInputText;" in core_source

    assert "sendBtn.disabled=sendDisabled;" in core_source
    assert "sendDropdownBtn.disabled=sendDisabled;" in core_source
    assert "modeQueueBtn.disabled = sendDisabled;" in core_source
    assert "modeSteerBtn.disabled = sendDisabled;" in core_source


def test_chat_mode_controls_and_send_label_are_inflight_only() -> None:
    core_source = Path("orchestrator/web/static/app-core.js").read_text(encoding="utf-8")

    assert "const showModeControls = isRunning || S.chatStopPending;" in core_source
    assert "sendDropdownBtn.classList.toggle('hidden', !showModeControls);" in core_source
    assert "sendBtn.textContent = mode==='steer' ? 'Send to Steer' : 'Send to Queue';" in core_source
    assert "sendBtn.textContent='Send';" in core_source


def test_chat_stop_uses_terminal_state_fallback_and_reconcile() -> None:
    events_source = Path("orchestrator/web/static/app-events.js").read_text(encoding="utf-8")
    ws_source = Path("orchestrator/web/static/app-ws.js").read_text(encoding="utf-8")

    assert "state: 'cancelled'" in events_source
    assert "source: 'local_timeout'" in events_source
    assert "type:'chat_request_reconcile'" in events_source

    assert "S.chatTerminalStateByRequest[rid] = {state:'superseded', reason:'Stopped by user'" in ws_source
    assert "S.chatTerminalStateByRequest[rid] = {state, reason, ts:Date.now(), source:'chat_run_state'};" in ws_source
