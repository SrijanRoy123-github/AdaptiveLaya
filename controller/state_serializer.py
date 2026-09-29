from __future__ import annotations

import json
from typing import Any


def build_state(
    record: dict[str, Any],
    remaining_tokens: int,
    remaining_tools: int,
    remaining_latency_ms: int,
) -> dict[str, Any]:
    return {
        "user_request": str(record.get("user_request") or ""),
        "repository_summary": str(record.get("repository_summary") or ""),
        "relevant_file_context": str(record.get("relevant_file_context") or ""),
        "recent_changes": str(record.get("recent_changes") or ""),
        "previous_actions": [str(a) for a in (record.get("previous_actions") or [])],
        "test_results": str(record.get("test_results") or ""),
        "previous_tool_result": str(record.get("previous_tool_result") or ""),
        "remaining_token_budget": int(remaining_tokens),
        "remaining_tool_budget": int(remaining_tools),
        "remaining_latency_budget_ms": int(remaining_latency_ms),
        "step_index": int(record.get("step_index", 0)),
        "previous_confidence": float(record.get("previous_confidence", 0.0)),
    }


def budget_vector(state: dict[str, Any], regime: dict[str, Any]) -> list[float]:
    max_tokens = max(int(regime["max_tokens"]), 1)
    max_tools = max(int(regime["max_tools"]), 1)
    max_latency = max(int(regime["max_latency_ms"]), 1)
    max_steps = max(int(regime["max_steps"]), 1)
    return [
        min(max(state["remaining_token_budget"] / max_tokens, 0.0), 1.0),
        min(max(state["remaining_tool_budget"] / max_tools, 0.0), 1.0),
        min(max(state["remaining_latency_budget_ms"] / max_latency, 0.0), 1.0),
        min(max(state["step_index"] / max_steps, 0.0), 1.0),
    ]


def serialize_state(
    state: dict[str, Any],
    regime: dict[str, Any] | None = None,
    max_tokens: int = 512,
) -> str:
    regime = regime or {"max_tokens": 1, "max_tools": 1, "max_latency_ms": 1, "max_steps": 1}
    b = budget_vector(state, regime)
    b_str = ",".join(f"{x:.2f}" for x in b)

    def cap(value: Any, limit: int = 80) -> str:
        return " ".join(str(value).split()[:limit])

    summary_line = (
        f"step={state['step_index']} tok={state['remaining_token_budget']} "
        f"tools={state['remaining_tool_budget']} lat_ms={state['remaining_latency_budget_ms']} "
        f"conf={state['previous_confidence']:.2f} b=[{b_str}]"
    )
    lines = [
        summary_line,
        f"REQ: {cap(state['user_request'])}",
        f"REPO: {cap(state['repository_summary'])}",
        f"CTX: {cap(state['relevant_file_context'])}",
        f"CHANGES: {cap(state['recent_changes'])}",
        f"PREV_ACTIONS: {' '.join(state['previous_actions']) or 'none'}",
        f"TESTS: {cap(state['test_results']) or 'none'}",
        f"PREV_OBS: {cap(state['previous_tool_result']) or 'none'}",
    ]
    kept: list[str] = []
    count = 0
    for line in lines:
        line_words = len(line.split())
        if kept and count + line_words > max_tokens:
            break
        kept.append(line)
        count += line_words
    text = "\n".join(kept)
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])
    return text


def state_to_json(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True)
