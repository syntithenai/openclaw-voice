from orchestrator.gateway.quick_answer import sanitize_quick_answer_text


def test_sanitize_quick_answer_text_strips_asterisk_markdown() -> None:
    text = "This is **bold** and *italic*"
    assert sanitize_quick_answer_text(text) == "This is bold and italic"


def test_sanitize_quick_answer_text_collapses_whitespace_after_strip() -> None:
    text = "Keep  **two**   spaces *clean*"
    assert sanitize_quick_answer_text(text) == "Keep two spaces clean"


def test_sanitize_quick_answer_text_handles_non_string() -> None:
    assert sanitize_quick_answer_text(None) == ""


def test_sanitize_quick_answer_text_extracts_nested_tool_result_response() -> None:
    payload = {
        "success": True,
        "result": {
            "alarm_id": "123",
            "trigger_time": 1773234000.0,
            "label": "",
            "response": "Alarm set for 12:00 AM",
        },
    }
    assert sanitize_quick_answer_text(payload) == "Alarm set for 12:00 AM"


def test_sanitize_quick_answer_text_falls_back_to_label() -> None:
    payload = {
        "success": True,
        "result": {
            "alarm_id": "123",
            "label": "wake up",
        },
    }
    assert sanitize_quick_answer_text(payload) == "wake up"
