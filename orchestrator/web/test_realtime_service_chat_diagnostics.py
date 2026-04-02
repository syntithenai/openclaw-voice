from pathlib import Path


def test_chat_diagnostics_setting_routes_contract() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert 'if msg_type == "chat_verbose_set" and self._on_chat_verbose_set:' in source
    assert 'if msg_type == "chat_reasoning_set" and self._on_chat_reasoning_set:' in source
    assert 'if msg_type == "chat_lifecycle_policy_set" and self._on_chat_lifecycle_policy_set:' in source
    assert 'if msg_type == "chat_interim_set" and self._on_chat_interim_set:' in source
    assert '"type": "setting_action_ack"' in source
    assert '"type": "setting_action_error"' in source


def test_chat_lifecycle_visibility_channels_contract() -> None:
    source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "if not hasattr(gateway, \"listen_raw\") and _debug_log_enabled():" in source
    assert "if step_msg and _chat_detail_enabled():" in source
    assert "if interim_msg and _chat_detail_enabled() and web_chat_interim_enabled:" in source
