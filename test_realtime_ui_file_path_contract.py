from pathlib import Path


UI_SOURCE = "\n".join(
    Path(path).read_text(encoding="utf-8")
    for path in (
        "orchestrator/web/static/app-core.js",
        "orchestrator/web/static/app-events.js",
        "orchestrator/web/static/app-render.js",
        "orchestrator/web/static/app-ws.js",
    )
)


def test_tool_request_extracts_snake_case_file_path() -> None:
    assert "req.filePath||req.file_path||req.path||req.old_path||req.new_path||req.uri" in UI_SOURCE


def test_thinking_block_shows_waiting_icon_in_summary() -> None:
    assert "const thinkingSummary=waiting" in UI_SOURCE
    assert "animate-spin" in UI_SOURCE


def test_exec_preview_clamped_to_two_lines() -> None:
    assert "const clampPreviewLines=(raw, maxLines=2)=>" in UI_SOURCE
    assert "clampPreviewLines(execCommand, 2)" in UI_SOURCE


def test_transient_lifecycle_errors_not_auto_terminal_failure() -> None:
    assert "const isTransientLifecycleError=(phase, errText)=>" in UI_SOURCE
    assert "const hasLifecycleError=hasLifecycleHardError" in UI_SOURCE
