import json
from pathlib import Path

import yaml

from controller.state_serializer import budget_vector, build_state, serialize_state

ROOT = Path(__file__).resolve().parents[1]


def sample_state(**over):
    base = {
        "user_request": "Fix the auth test failure.",
        "repository_summary": "FastAPI app, pytest suite, 3 files.",
        "relevant_file_context": "def check_token(): ...",
        "recent_changes": "auth.py modified",
        "previous_actions": [],
        "test_results": "",
        "previous_tool_result": "",
        "remaining_token_budget": 750,
        "remaining_tool_budget": 2,
        "remaining_latency_budget_ms": 60000,
        "step_index": 0,
        "previous_confidence": 0.0,
    }
    base.update(over)
    return base


def test_serialize_contains_all_fields():
    text = serialize_state(sample_state())
    for needle in ["Fix the auth test failure.", "step=0", "tok=750", "tools=2", "lat_ms=60000", "conf=0.00"]:
        assert needle in text


def test_serialize_budget_vector_normalized():
    regimes = yaml.safe_load((ROOT / "configs" / "regimes.yaml").read_text(encoding="utf-8"))["regimes"]
    text = serialize_state(sample_state(), regimes["NORMAL"])
    assert "b=[" in text


def test_serialize_within_token_budget():
    state = sample_state(repository_summary="word " * 2000)
    text = serialize_state(state, max_tokens=512)
    assert len(text.split()) <= 512
    assert "step=" in text
    assert "b=[" in text


def test_build_state_from_record():
    record = {
        "user_request": "Add login endpoint",
        "repository_summary": "api/",
        "relevant_file_context": "main.py",
        "recent_changes": "",
        "previous_actions": ["TOOL"],
        "test_results": "ok",
        "previous_tool_result": "READ_CONTEXT done",
        "step_index": 2,
        "previous_confidence": 0.4,
    }
    state = build_state(record, remaining_tokens=700, remaining_tools=1, remaining_latency_ms=50000)
    assert state["remaining_token_budget"] == 700
    assert state["remaining_tool_budget"] == 1
    assert state["step_index"] == 2
    json.dumps(state)
    import jsonschema
    schema = json.loads((ROOT / "schemas" / "state_schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(state, schema)


def test_budget_vector_range():
    regime = {"max_tokens": 100, "max_tools": 2, "max_latency_ms": 1000, "max_steps": 4}
    v = budget_vector(sample_state(remaining_token_budget=50, remaining_tool_budget=1,
                                   remaining_latency_budget_ms=500, step_index=2), regime)
    assert len(v) == 4
    assert all(0.0 <= x <= 1.0 for x in v)
    assert v == [0.5, 0.5, 0.5, 0.5]
    over = budget_vector(sample_state(remaining_token_budget=500, remaining_tool_budget=9,
                                      remaining_latency_budget_ms=9000, step_index=10), regime)
    assert all(x == 1.0 for x in over)
    neg = budget_vector(sample_state(remaining_token_budget=-5, remaining_tool_budget=-1,
                                     remaining_latency_budget_ms=-100, step_index=0), regime)
    assert all(x == 0.0 for x in neg)
