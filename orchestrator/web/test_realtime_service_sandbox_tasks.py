from pathlib import Path


def test_sandbox_task_snapshot_and_broadcast_contract() -> None:
    source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert '"sandbox_tasks": list(self._sandbox_tasks.values())' in source
    assert 'def update_sandbox_exec_task(self, task: dict[str, Any]) -> None:' in source
    assert '"type": "sandbox_exec_update"' in source
    assert 'def append_sandbox_exec_log(' in source
    assert '"type": "sandbox_exec_log_append"' in source
    assert 'if msg_type == "sandbox_task_logs_get":' in source
    assert '"type": "sandbox_task_logs"' in source
