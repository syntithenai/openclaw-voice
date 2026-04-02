import asyncio
from pathlib import Path

import pytest

from orchestrator.web.realtime_service import EmbeddedVoiceWebService


@pytest.mark.asyncio
async def test_sandbox_exec_task_updates_and_logs_broadcast(tmp_path: Path) -> None:
    service = EmbeddedVoiceWebService(
        auth_mode="disabled",
        chat_persist_path=str(tmp_path / "chat_state.json"),
    )

    broadcasts: list[dict] = []

    async def _fake_broadcast(payload: dict) -> None:
        broadcasts.append(dict(payload))

    service.broadcast = _fake_broadcast  # type: ignore[method-assign]

    service.update_sandbox_exec_task(
        {
            "task_id": "exec-123",
            "container_name": "sandbox-a",
            "exec_id": "e-1",
            "status": "running",
            "command": "echo hello",
        }
    )
    service.append_sandbox_exec_log("exec-123", ["hello"], exec_id="e-1", stream="stdout", seq=7)
    await asyncio.sleep(0)

    assert service._sandbox_tasks["exec-123"]["status"] == "running"
    assert service._sandbox_tasks["exec-123"]["task_type"] == "sandbox"
    assert service._sandbox_task_logs["exec-123"][-1]["seq"] == 7
    assert service._sandbox_task_logs["exec-123"][-1]["stream"] == "stdout"

    msg_types = [msg.get("type") for msg in broadcasts]
    assert "sandbox_exec_update" in msg_types
    assert "sandbox_exec_log_append" in msg_types


@pytest.mark.asyncio
async def test_subagent_tracking_terminal_and_thinking_broadcast(tmp_path: Path) -> None:
    service = EmbeddedVoiceWebService(
        auth_mode="disabled",
        chat_persist_path=str(tmp_path / "chat_state.json"),
    )

    broadcasts: list[dict] = []

    async def _fake_broadcast(payload: dict) -> None:
        broadcasts.append(dict(payload))

    service.broadcast = _fake_broadcast  # type: ignore[method-assign]

    service.update_subagent_task({"run_id": "req-9", "status": "running", "step": "tool:start"})
    service.append_subagent_thinking("req-9", "planning...", seq=3)
    service.mark_subagent_terminal("req-9", "completed", summary="done")
    await asyncio.sleep(0)

    assert service._subagent_tasks["req-9"]["status"] == "completed"
    assert service._subagent_tasks["req-9"]["summary"] == "done"
    assert service._subagent_thinking_logs["req-9"][-1]["seq"] == 3

    msg_types = [msg.get("type") for msg in broadcasts]
    assert "subagent_task_update" in msg_types
    assert "subagent_thinking_append" in msg_types
    assert "subagent_task_terminal" in msg_types


@pytest.mark.asyncio
async def test_task_tracking_present_in_state_snapshot_contract(tmp_path: Path) -> None:
    service = EmbeddedVoiceWebService(
        auth_mode="disabled",
        chat_persist_path=str(tmp_path / "chat_state.json"),
    )
    service.update_sandbox_exec_task({"task_id": "exec-xyz", "status": "completed"})
    service.update_subagent_task({"run_id": "req-xyz", "status": "failed"})
    await asyncio.sleep(0)

    snapshot = service._build_state_snapshot()
    assert snapshot["type"] == "state_snapshot"
    assert any(item.get("task_id") == "exec-xyz" for item in snapshot["sandbox_tasks"])
    assert any(item.get("task_id") == "req-xyz" for item in snapshot["subagent_tasks"])


def test_task_log_backfill_action_handlers_exist() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert 'if msg_type == "sandbox_task_logs_get":' in source
    assert '"type": "sandbox_task_logs"' in source
    assert 'if msg_type == "subagent_task_thinking_get":' in source
    assert '"type": "subagent_task_thinking"' in source
