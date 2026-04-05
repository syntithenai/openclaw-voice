from __future__ import annotations

import time
from typing import Any


_TOOLISH_ROLES = {"tool", "toolresult", "tool_result", "tool_use", "tooluse"}
_TOOL_CALL_TYPES = {"toolcall", "tool_call", "tool_use", "tooluse"}
_TOOL_RESULT_TYPES = {"toolresult", "tool_result", "tool_result_error"}
_THINKING_TYPES = {"thinking", "reasoning"}
_TEXT_BLOCK_TYPES = {"text", "output_text", "input_text", "markdown"}
_TRANSCRIPT_EVENT_TYPES = {"session", "model_change", "thinking_level_change", "custom", "compaction"}


def _flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        ctype = str(item.get("type") or "").strip().lower()
        if ctype in {"thinking", "reasoning", "toolcall", "toolresult"}:
            continue
        text_val = item.get("text")
        if not isinstance(text_val, str) and ctype in _TEXT_BLOCK_TYPES:
            text_val = item.get("value")
        if isinstance(text_val, str) and text_val.strip():
            parts.append(text_val.strip())
    return "\n".join(parts).strip()


def _normalize_block_type(block: dict[str, Any]) -> str:
    return str(block.get("type") or "").strip().lower()


def _next_id(seq: int) -> tuple[int, int]:
    seq += 1
    return seq, seq


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def _json_details(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _tool_summary_text(block: dict[str, Any]) -> str:
    for key in ("result", "partialResult", "output", "stdout", "stderr", "message", "text", "content"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    text_val = item.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        parts.append(text_val.strip())
            if parts:
                return "\n".join(parts).strip()
    return ""


def _map_content_block_to_voice_messages(
    *,
    block: dict[str, Any],
    message_obj: dict[str, Any],
    role: str,
    ts: float,
    source: str,
    request_id: str,
    mapped: list[dict[str, Any]],
    seq: int,
) -> int:
    block_type = _normalize_block_type(block)
    if block_type in _THINKING_TYPES:
        thinking_text = _stringify(block.get("thinking") or block.get("text"))
        if not thinking_text:
            return seq
        seq, msg_id = _next_id(seq)
        mapped.append(
            {
                "id": msg_id,
                "role": "interim",
                "text": "reasoning",
                "phase": "update",
                "details": _json_details({"text": thinking_text, "type": block_type}),
                "ts": ts,
                "source": source,
                "request_id": request_id,
            }
        )
        return seq

    if block_type in _TOOL_CALL_TYPES:
        tool_name = _stringify(block.get("name") or role or "tool") or "tool"
        tool_call_id = _stringify(
            block.get("id")
            or block.get("toolCallId")
            or block.get("tool_call_id")
            or message_obj.get("toolCallId")
            or message_obj.get("tool_call_id")
        )
        phase = _stringify(block.get("phase") or "start") or "start"
        seq, msg_id = _next_id(seq)
        mapped.append(
            {
                "id": msg_id,
                "role": "step",
                "text": tool_name,
                "name": tool_name,
                "phase": phase,
                "tool_call_id": tool_call_id,
                "details": _json_details(block),
                "ts": ts,
                "source": source,
                "request_id": request_id,
            }
        )
        return seq

    if block_type in _TOOL_RESULT_TYPES:
        tool_name = _stringify(block.get("name") or message_obj.get("toolName") or message_obj.get("tool_name") or role or "tool") or "tool"
        tool_call_id = _stringify(
            block.get("toolCallId")
            or block.get("tool_call_id")
            or block.get("id")
            or message_obj.get("toolCallId")
            or message_obj.get("tool_call_id")
        )
        phase = "error" if block_type.endswith("_error") or block.get("is_error") is True else "result"
        payload = dict(block)
        summary = _tool_summary_text(block)
        if summary and "result" not in payload and "text" not in payload:
            payload["result"] = summary
        seq, msg_id = _next_id(seq)
        mapped.append(
            {
                "id": msg_id,
                "role": "step",
                "text": tool_name,
                "name": tool_name,
                "phase": phase,
                "tool_call_id": tool_call_id,
                "details": _json_details(payload),
                "ts": ts,
                "source": source,
                "request_id": request_id,
            }
        )
        return seq

    return seq


def _normalize_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            return ts / 1000.0
        return ts
    if isinstance(value, str):
        try:
            return _normalize_ts(float(value))
        except Exception:
            return time.time()
    return time.time()


def _extract_request_id(message_obj: dict[str, Any], item: dict[str, Any]) -> str:
    for key in ("request_id", "requestId", "run_id", "runId"):
        value = message_obj.get(key)
        if value is not None:
            rid = _stringify(value)
            if rid:
                return rid
    parent_id = _stringify(item.get("parentId") or item.get("parent_id"))
    if parent_id:
        return parent_id
    return ""


def _build_full_text_content(message_obj: dict[str, Any]) -> str:
    content = message_obj.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return _stringify(message_obj.get("text"))

    sections: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = _normalize_block_type(block)
        if block_type in _THINKING_TYPES:
            reasoning = _stringify(block.get("thinking") or block.get("text"))
            if reasoning:
                sections.append(f"Reasoning:\n{reasoning}")
            continue
        if block_type in _TOOL_CALL_TYPES:
            tool_name = _stringify(block.get("name") or "tool") or "tool"
            sections.append(f"Tool call ({tool_name}):\n{_json_details(block)}")
            continue
        if block_type in _TOOL_RESULT_TYPES:
            summary = _tool_summary_text(block)
            if summary:
                sections.append(f"Tool result:\n{summary}")
            else:
                sections.append(f"Tool result:\n{_json_details(block)}")
            continue

        text_val = _stringify(block.get("text") or block.get("value"))
        if text_val:
            sections.append(text_val)
            continue
        sections.append(_json_details(block))

    return "\n\n".join(s for s in sections if s).strip()


def _map_transcript_event(
    *,
    item: dict[str, Any],
    item_type: str,
    ts: float,
    source: str,
    seq: int,
) -> tuple[int, dict[str, Any] | None]:
    if item_type not in _TRANSCRIPT_EVENT_TYPES:
        return seq, None

    phase_by_type = {
        "session": "session",
        "model_change": "model_change",
        "thinking_level_change": "thinking_level_change",
        "custom": "custom",
        "compaction": "compaction",
    }
    label_by_type = {
        "session": "session metadata",
        "model_change": "model changed",
        "thinking_level_change": "thinking level changed",
        "custom": "custom event",
        "compaction": "compaction marker",
    }

    seq, msg_id = _next_id(seq)
    mapped = {
        "id": msg_id,
        "role": "interim",
        "text": label_by_type.get(item_type, item_type.replace("_", " ")),
        "phase": phase_by_type.get(item_type, item_type),
        "details": _json_details(item),
        "ts": ts,
        "source": source,
        "request_id": "",
    }
    return seq, mapped


def map_gateway_messages_to_voice_format(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map gateway chat.history items to the web UI chat shape.

    chat.history strips the JSONL envelope so each item is already the inner
    {role, content, timestamp} object.  We also handle the raw JSONL envelope
    shape {type:"message", message:{...}} in case the format ever changes.
    """
    mapped: list[dict[str, Any]] = []
    seq = 0

    for item in messages:
        if not isinstance(item, dict):
            continue

        # If still in JSONL envelope form: {type:"message", message:{role,...}}
        item_type = str(item.get("type") or "").strip().lower()
        if item_type in _TRANSCRIPT_EVENT_TYPES:
            ts = _normalize_ts(item.get("timestamp"))
            seq, event_msg = _map_transcript_event(
                item=item,
                item_type=item_type,
                ts=ts,
                source="gateway",
                seq=seq,
            )
            if event_msg is not None:
                mapped.append(event_msg)
            continue

        if item_type == "message" and isinstance(item.get("message"), dict):
            message_obj = item["message"]
        elif item_type in {"", "compaction"} or item_type == "message":
            # Already unwrapped by readSessionMessages; treat item itself as message obj.
            # Skip non-message JSONL types (session, model_change, etc.) that have no role.
            message_obj = item
        else:
            # Unknown envelope type — skip
            continue
        role = str(message_obj.get("role") or "").strip().lower()
        if not role:
            continue

        ts = _normalize_ts(message_obj.get("timestamp") or item.get("timestamp"))
        source = "gateway"
        request_id = _extract_request_id(message_obj, item)

        # Top-level tool messages can be valid transcript turns in some providers.
        if role in _TOOLISH_ROLES:
            tool_name = _stringify(
                message_obj.get("name") or message_obj.get("toolName") or message_obj.get("tool_name") or "tool"
            ) or "tool"
            tool_call_id = _stringify(
                message_obj.get("toolCallId") or message_obj.get("tool_call_id") or message_obj.get("id")
            )
            payload = dict(message_obj)
            summary = _tool_summary_text(message_obj)
            if summary and "text" not in payload and "result" not in payload:
                payload["text"] = summary
            seq, msg_id = _next_id(seq)
            mapped.append(
                {
                    "id": msg_id,
                    "role": "step",
                    "text": tool_name,
                    "name": tool_name,
                    "phase": "result",
                    "tool_call_id": tool_call_id,
                    "details": _json_details(payload),
                    "ts": ts,
                    "source": source,
                    "request_id": request_id,
                }
            )
            continue

        content = message_obj.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                seq = _map_content_block_to_voice_messages(
                    block=block,
                    message_obj=message_obj,
                    role=role,
                    ts=ts,
                    source=source,
                    request_id=request_id,
                    mapped=mapped,
                    seq=seq,
                )

        text = _flatten_text_content(message_obj.get("content"))
        if not text:
            text = str(message_obj.get("text") or "").strip()
        if not text:
            continue

        full_text = _build_full_text_content(message_obj)
        segment_kind = _stringify(message_obj.get("segment_kind") or message_obj.get("segmentKind") or "final") or "final"

        seq, msg_id = _next_id(seq)
        mapped.append(
            {
                "id": msg_id,
                "role": role,
                "text": text,
                "full_text": full_text if full_text else text,
                "segment_kind": segment_kind,
                "ts": ts,
                "source": source,
                "request_id": request_id,
            }
        )

    return mapped
