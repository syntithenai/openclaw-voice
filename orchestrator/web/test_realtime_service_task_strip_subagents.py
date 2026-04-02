from pathlib import Path


def test_subagent_task_snapshot_and_broadcast_contract() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert '"subagent_tasks": list(self._subagent_tasks.values())' in source
    assert 'def update_subagent_task(self, task: dict[str, Any]) -> None:' in source
    assert '"type": "subagent_task_update"' in source
    assert 'def append_subagent_thinking(' in source
    assert '"type": "subagent_thinking_append"' in source
    assert 'def mark_subagent_terminal(self, run_id: str, status: str, summary: str = "", error: str = "") -> None:' in source
    assert '"type": "subagent_task_terminal"' in source
    assert 'if msg_type == "subagent_task_thinking_get":' in source
