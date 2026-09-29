import json
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_json(name):
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def valid_state():
    return {
        "user_request": "Fix the failing test in auth.py",
        "repository_summary": "Small FastAPI app with pytest.",
        "relevant_file_context": "def check(): ...",
        "recent_changes": "none",
        "previous_actions": ["ANSWER"],
        "test_results": "1 failed",
        "previous_tool_result": "",
        "remaining_token_budget": 750,
        "remaining_tool_budget": 2,
        "remaining_latency_budget_ms": 30000,
        "step_index": 1,
        "previous_confidence": 0.6,
    }


def test_state_schema_valid():
    jsonschema.validate(valid_state(), load_json("state_schema.json"))


def test_state_schema_rejects_missing_field():
    state = valid_state()
    del state["step_index"]
    try:
        jsonschema.validate(state, load_json("state_schema.json"))
    except jsonschema.exceptions.ValidationError:
        return
    raise AssertionError("expected validation error")


def test_action_schema_tool_required_for_tool():
    schema = load_json("action_schema.json")
    jsonschema.validate(
        {"action": "TOOL", "tool": "RUN_TESTS", "terminal": "CONTINUE", "confidence": 0.8},
        schema,
    )
    try:
        jsonschema.validate(
            {"action": "JUMP", "tool": None, "terminal": "CONTINUE", "confidence": 0.5}, schema
        )
    except jsonschema.exceptions.ValidationError:
        return
    raise AssertionError("expected validation error")


def test_action_schema_tool_conditional():
    schema = load_json("action_schema.json")
    try:
        jsonschema.validate(
            {"action": "TOOL", "terminal": "CONTINUE", "confidence": 0.8}, schema
        )
    except jsonschema.exceptions.ValidationError:
        pass
    else:
        raise AssertionError("TOOL without tool must fail")

    try:
        jsonschema.validate(
            {"action": "ANSWER", "tool": "RUN_TESTS", "terminal": "CONTINUE", "confidence": 0.8},
            schema,
        )
    except jsonschema.exceptions.ValidationError:
        pass
    else:
        raise AssertionError("ANSWER with tool must fail")


def test_tool_schema():
    schema = load_json("tool_schema.json")
    jsonschema.validate(
        {"tool": "RUN_TESTS", "args": {"target": "tests/test_auth.py", "timeout_s": 60}}, schema
    )
    try:
        jsonschema.validate({"tool": "SHELL", "args": {}}, schema)
    except jsonschema.exceptions.ValidationError:
        return
    raise AssertionError("expected validation error")


def test_budget_schema():
    jsonschema.validate(
        {"max_steps": 8, "max_tool_calls": 3, "max_generated_tokens": 1000,
         "max_observation_tokens": 1000, "max_wall_time_ms": 600000},
        load_json("budget_schema.json"),
    )


def test_utility_config_shape():
    cfg = yaml.safe_load((ROOT / "configs" / "utility_config.yaml").read_text(encoding="utf-8"))
    assert {"utility", "cost_ledger"} <= set(cfg)
    assert cfg["utility"]["success_reward"] == 1.0
    for key in ("alpha_tool_calls", "beta_router_forwards", "gamma_wall_ms"):
        assert key in cfg["cost_ledger"]


def test_regimes_config():
    cfg = yaml.safe_load((ROOT / "configs" / "regimes.yaml").read_text(encoding="utf-8"))
    assert set(cfg["regimes"]) == {"LOW", "NORMAL", "HIGH"}
    assert cfg["regimes"]["LOW"]["max_tokens"] == 250
    assert cfg["regimes"]["NORMAL"]["max_tokens"] == 750
    assert cfg["regimes"]["HIGH"]["max_tokens"] == 1500
    assert cfg["regimes"]["LOW"]["max_tools"] == 1
    assert cfg["regimes"]["NORMAL"]["max_tools"] == 2
    assert cfg["regimes"]["HIGH"]["max_tools"] == 4
    assert cfg["regimes"]["LOW"]["max_latency_ms"] == 20000
    assert cfg["regimes"]["LOW"]["max_steps"] == 4
    assert cfg["regimes"]["NORMAL"]["max_latency_ms"] == 60000
    assert cfg["regimes"]["NORMAL"]["max_steps"] == 8
    assert cfg["regimes"]["HIGH"]["max_latency_ms"] == 120000
    assert cfg["regimes"]["HIGH"]["max_steps"] == 12


def test_worker_config_matrix():
    cfg = yaml.safe_load((ROOT / "configs" / "worker.yaml").read_text(encoding="utf-8"))
    ids = {w["id"]: w for w in cfg["workers"]}
    assert set(ids) == {0, 1}
    assert ids[0]["seed"] == 42 and ids[0]["specialization"] == "swe_repair"
    assert ids[1]["seed"] == 43 and ids[1]["specialization"] == "codegen"


def test_aggregation_config():
    cfg = yaml.safe_load((ROOT / "configs" / "aggregation.yaml").read_text(encoding="utf-8"))
    assert cfg["aggregation"]["method"] in {"best_worker", "equal_weight", "validation_weighted"}
    assert cfg["aggregation"]["min_workers"] == 2
    assert cfg["aggregation"]["rounds"] == 2


def test_phase_configs_parse():
    for name in ["phase0_qwen", "phase0_encoder", "phase1_mvp", "phase2_lora"]:
        cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert "phase" in cfg
    gates = yaml.safe_load((ROOT / "configs" / "phase0_qwen.yaml").read_text(encoding="utf-8"))["eval"]["gates"]
    assert set(gates) == {"macro_f1", "hard_negative_accuracy", "ece_max"}
