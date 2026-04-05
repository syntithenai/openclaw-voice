from orchestrator.gateway.session_mapper import map_gateway_messages_to_voice_format


def test_history_mapper_preserves_thinking_and_tool_timeline_blocks() -> None:
    mapped = map_gateway_messages_to_voice_format(
        [
            {
                "role": "user",
                "timestamp": 1,
                "content": [{"type": "text", "text": "Find the latest logs"}],
            },
            {
                "role": "assistant",
                "timestamp": 2,
                "content": [
                    {"type": "thinking", "thinking": "Checking recent files"},
                    {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "/tmp/log"}},
                    {"type": "text", "text": "I found the latest logs."},
                ],
            },
            {
                "role": "toolResult",
                "timestamp": 3,
                "toolCallId": "call_1",
                "content": [{"type": "tool_result", "text": "Read 120 lines", "name": "read"}],
            },
        ]
    )

    assert [msg["role"] for msg in mapped] == ["user", "interim", "step", "assistant", "step"]
    assert mapped[1]["text"] == "reasoning"
    assert 'Checking recent files' in str(mapped[1]["details"])
    assert mapped[2]["name"] == "read"
    assert mapped[2]["tool_call_id"] == "call_1"
    assert mapped[3]["text"] == "I found the latest logs."
    assert mapped[4]["phase"] == "result"
    assert mapped[4]["tool_call_id"] == "call_1"
    assert 'Read 120 lines' in str(mapped[4]["details"])


def test_history_mapper_ignores_hidden_tool_only_turn_without_result_text() -> None:
    mapped = map_gateway_messages_to_voice_format(
        [
            {
                "role": "assistant",
                "timestamp": 2,
                "content": [
                    {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "/tmp/log"}},
                ],
            }
        ]
    )

    assert len(mapped) == 1
    assert mapped[0]["role"] == "step"
    assert mapped[0]["name"] == "read"


def test_history_mapper_preserves_transcript_events_as_interim_rows() -> None:
    mapped = map_gateway_messages_to_voice_format(
        [
            {"type": "session", "timestamp": 1, "id": "sess-1", "version": 1},
            {"type": "model_change", "timestamp": 2, "modelId": "claude-sonnet-4-6"},
            {
                "role": "user",
                "timestamp": 3,
                "content": [{"type": "text", "text": "hello"}],
            },
        ]
    )

    assert [msg["role"] for msg in mapped] == ["interim", "interim", "user"]
    assert mapped[0]["phase"] == "session"
    assert mapped[1]["phase"] == "model_change"
    assert '"modelId": "claude-sonnet-4-6"' in str(mapped[1]["details"])


def test_history_mapper_populates_full_text_for_assistant_blocks() -> None:
    mapped = map_gateway_messages_to_voice_format(
        [
            {
                "role": "assistant",
                "timestamp": 10,
                "content": [
                    {"type": "thinking", "thinking": "Plan steps"},
                    {"type": "text", "text": "Short spoken summary."},
                    {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "README.md"}},
                ],
            }
        ]
    )

    assistant = next(msg for msg in mapped if msg["role"] == "assistant")
    assert assistant["text"] == "Short spoken summary."
    assert "Reasoning:" in str(assistant.get("full_text", ""))
    assert "Tool call (read):" in str(assistant.get("full_text", ""))