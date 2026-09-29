# Adaptive Laya — Part 1: Core Foundations + Phase-0 Data (Tasks 0–8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the repo and deliver spec §45 artifacts 1–10: schemas, configs, state serializer, cost ledger, budget/loop guards, router, losses/calibration, frozen backbone, and the seed decision dataset generator with hard negatives + counterfactual labels.

**Architecture:** Frozen pretrained decoder exposes a hidden state at `<|route|>`; a 1–5M-param router (LayerNorm → MLP → 4 heads) predicts ANSWER/TOOL/ASK + tool + confidence + terminal. Policy state is a validated JSON dict serialized to ≤512 tokens with a normalized 4-dim budget vector. Data is synthetic-seed JSONL with manifest checksums.

**Tech Stack:** Python 3.10+, PyTorch 2.x, transformers, jsonschema, PyYAML, pytest, ruff.

**Spec:** `Adaptive_Laya_600M_Three_Kaggle_Accounts_Final_Plan.md`.

**Locked decisions:** default backbone `Qwen/Qwen2.5-Coder-0.5B-Instruct` (spec §3's 1.5B conflicts with the ~600M total in §100/§103; 1.5B is config-switchable under a different experiment name). Encoder probe uses configurable `AutoModel` (default `google-bert/bert-base-multilingual-cased`).

**Companion plans:** Part 2 (Tasks 9–15: sharding, probes, train/eval, tools, agent loop), Part 3 (Tasks 16–23: LoRA worker, 3-account aggregation, Kaggle notebooks + runbook).

---

## File Structure (Part 1 scope)

```
adaptive-laya/
├── requirements.txt
├── pytest.ini
├── schemas/{state,action,tool,budget}_schema.json
├── configs/{utility_config,regimes,worker,aggregation,phase0_qwen,phase0_encoder,phase1_mvp,phase2_lora}.yaml
├── model/{__init__,routing_heads,router,losses,calibration,backbone}.py
├── controller/{__init__,state_serializer,cost_ledger,budget_manager,loop_guard}.py
├── scripts/data_generation/{__init__,generate_decisions,hard_negatives,counterfactuals,perturb}.py
├── data/{raw,processed,decisions,counterfactuals,trajectories}/.gitkeep
├── results/.gitkeep
└── tests/{test_schemas,test_state_serializer,test_cost_ledger,test_budget_guard,test_router,test_losses_calibration,test_backbone,test_data_generation}.py
```

**Import rule:** `model/` and `controller/` import nothing from `scripts/`; `scripts/data_generation` imports nothing from `model/`/`controller/`. Tests import with repo root on `pythonpath` via `pytest.ini`.

---

### Task 0: Repository scaffold

**Files:**
- Create: `requirements.txt`, `pytest.ini`, package `__init__.py` files, `data/*/.gitkeep`, `results/.gitkeep`

- [ ] **Step 1: Initialize repo and skeleton**

```powershell
git init
New-Item -ItemType Directory -Force -Path model,controller,tools,scripts,scripts\data_generation,scripts\training,scripts\evaluation,data\raw,data\processed,data\decisions,data\counterfactuals,data\trajectories,results,configs,schemas,kaggle,tests | Out-Null
New-Item -ItemType File -Force -Path model\__init__.py,controller\__init__.py,tools\__init__.py,scripts\__init__.py,scripts\data_generation\__init__.py,scripts\training\__init__.py,scripts\evaluation\__init__.py,data\raw\.gitkeep,data\processed\.gitkeep,data\decisions\.gitkeep,data\counterfactuals\.gitkeep,data\trajectories\.gitkeep,results\.gitkeep | Out-Null
```

- [ ] **Step 2: Write `requirements.txt`**

```text
torch>=2.1
transformers>=4.44
peft>=0.12
datasets>=2.20
accelerate>=0.33
jsonschema>=4.22
PyYAML>=6.0
numpy>=1.26
tqdm>=4.66
ruff>=0.5
kaggle>=1.6
kagglehub>=0.2
pytest>=8.0
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -q
```

- [ ] **Step 4: Install and verify**

Run: `pip install -r requirements.txt`
Expected: installs without error.

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "chore: scaffold adaptive-laya repo"
```

---

### Task 1: Schemas and configs (spec §45 items 1–5, §7, §14)

**Files:**
- Create: `schemas/state_schema.json`, `schemas/action_schema.json`, `schemas/tool_schema.json`, `schemas/budget_schema.json`
- Create: `configs/utility_config.yaml`, `configs/regimes.yaml`, `configs/worker.yaml`, `configs/aggregation.yaml`, `configs/phase0_qwen.yaml`, `configs/phase0_encoder.yaml`, `configs/phase1_mvp.yaml`, `configs/phase2_lora.yaml`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

```python
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


def test_worker_config_matrix():
    cfg = yaml.safe_load((ROOT / "configs" / "worker.yaml").read_text(encoding="utf-8"))
    ids = {w["id"]: w for w in cfg["workers"]}
    assert ids[0]["seed"] == 42 and ids[0]["specialization"] == "swe_repair"
    assert ids[1]["seed"] == 43 and ids[1]["specialization"] == "codegen"
    assert ids[2]["seed"] == 44 and ids[2]["specialization"] == "debug_tools"


def test_aggregation_config():
    cfg = yaml.safe_load((ROOT / "configs" / "aggregation.yaml").read_text(encoding="utf-8"))
    assert cfg["aggregation"]["method"] in {"best_worker", "equal_weight", "validation_weighted"}
    assert cfg["aggregation"]["min_workers"] == 2
    assert cfg["aggregation"]["rounds"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL — files not found.

- [ ] **Step 3: Write `schemas/state_schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "user_request", "repository_summary", "relevant_file_context",
    "recent_changes", "previous_actions", "test_results",
    "previous_tool_result", "remaining_token_budget",
    "remaining_tool_budget", "remaining_latency_budget_ms",
    "step_index", "previous_confidence"
  ],
  "additionalProperties": false,
  "properties": {
    "user_request": {"type": "string"},
    "repository_summary": {"type": "string"},
    "relevant_file_context": {"type": "string"},
    "recent_changes": {"type": "string"},
    "previous_actions": {"type": "array", "items": {"type": "string"}},
    "test_results": {"type": "string"},
    "previous_tool_result": {"type": "string"},
    "remaining_token_budget": {"type": "integer", "minimum": 0},
    "remaining_tool_budget": {"type": "integer", "minimum": 0},
    "remaining_latency_budget_ms": {"type": "integer", "minimum": 0},
    "step_index": {"type": "integer", "minimum": 0},
    "previous_confidence": {"type": "number", "minimum": 0, "maximum": 1}
  }
}
```

- [ ] **Step 4: Write `schemas/action_schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["action", "terminal", "confidence"],
  "additionalProperties": false,
  "properties": {
    "action": {"enum": ["ANSWER", "TOOL", "ASK"]},
    "tool": {"enum": ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY", null]},
    "terminal": {"enum": ["CONTINUE", "DONE", "ESCALATE"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "allOf": [
    {
      "if": {"properties": {"action": {"const": "TOOL"}}},
      "then": {"required": ["tool"], "properties": {"tool": {"enum": ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]}}}
    },
    {
      "if": {"properties": {"action": {"enum": ["ANSWER", "ASK"]}}},
      "then": {"properties": {"tool": {"enum": [null]}}}
    }
  ]
}
```

- [ ] **Step 5: Write `schemas/tool_schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["tool", "args"],
  "additionalProperties": false,
  "properties": {
    "tool": {"enum": ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]},
    "args": {
      "type": "object",
      "properties": {
        "target": {"type": "string"},
        "path": {"type": "string"},
        "pattern": {"type": "string"},
        "timeout_s": {"type": "integer", "minimum": 1, "maximum": 300}
      },
      "additionalProperties": true
    }
  }
}
```

- [ ] **Step 6: Write `schemas/budget_schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["max_steps", "max_tool_calls", "max_generated_tokens", "max_observation_tokens", "max_wall_time_ms"],
  "additionalProperties": false,
  "properties": {
    "max_steps": {"type": "integer", "minimum": 1},
    "max_tool_calls": {"type": "integer", "minimum": 0},
    "max_generated_tokens": {"type": "integer", "minimum": 1},
    "max_observation_tokens": {"type": "integer", "minimum": 1},
    "max_wall_time_ms": {"type": "integer", "minimum": 1000}
  }
}
```

- [ ] **Step 7: Write `configs/utility_config.yaml`**

```yaml
utility:
  success_reward: 1.0
  alpha_tokens: 2.0e-4
  beta_tool_calls: 5.0e-2
  gamma_latency_ms: 1.0e-5
  delta_router_forwards: 1.0e-3
cost_ledger:
  alpha_tool_calls: 5.0e-2
  beta_router_forwards: 1.0e-3
  gamma_wall_ms: 1.0e-5
```

- [ ] **Step 8: Write `configs/regimes.yaml`**

```yaml
regimes:
  LOW:
    max_tokens: 250
    max_tools: 1
    max_latency_ms: 20000
    max_steps: 4
  NORMAL:
    max_tokens: 750
    max_tools: 2
    max_latency_ms: 60000
    max_steps: 8
  HIGH:
    max_tokens: 1500
    max_tools: 4
    max_latency_ms: 120000
    max_steps: 12
```

- [ ] **Step 9: Write `configs/worker.yaml`**

```yaml
project:
  name: adaptive-laya-600m
datasets:
  decisions: adaptive-laya/adaptive-laya-decisions
  merged: adaptive-laya/adaptive-laya-merged
workers:
  - id: 0
    seed: 42
    shard_count: 3
    specialization: swe_repair
    artifact_slug: adaptive-laya/adaptive-laya-worker-0
  - id: 1
    seed: 43
    shard_count: 3
    specialization: codegen
    artifact_slug: adaptive-laya/adaptive-laya-worker-1
  - id: 2
    seed: 44
    shard_count: 3
    specialization: debug_tools
    artifact_slug: adaptive-laya/adaptive-laya-worker-2
shared_validation_fraction: 0.1
shared_test_fraction: 0.1
common_mixture: 0.7
specialization_fraction: 0.3
session:
  max_session_hours: 10.0
  publish_at_hours: 9.5
  checkpoint_every_steps: 500
```

- [ ] **Step 10: Write `configs/aggregation.yaml`**

```yaml
aggregation:
  method: equal_weight
  weight_by: examples
  validation_gate: true
  min_workers: 2
  rounds: 2
  experiments:
    - best_worker
    - equal_weight
    - validation_weighted
```

- [ ] **Step 11: Write phase configs**

`configs/phase0_qwen.yaml`:

```yaml
phase: 0
mode: probe
backbone:
  kind: decoder
  model_id: Qwen/Qwen2.5-Coder-0.5B-Instruct
  dtype: float16
  route_token: "<|route|>"
router:
  hidden_size: 896
  mlp_hidden: 512
  mlp_out: 256
training:
  lr: 3.0e-4
  epochs: 5
  batch_size: 64
  weight_decay: 0.01
  fp16: true
  seed: 42
eval:
  ece_bins: 15
  bootstrap_samples: 1000
  gates:
    macro_f1: 0.70
    hard_negative_accuracy: 0.75
    ece_max: 0.15
```

`configs/phase0_encoder.yaml`:

```yaml
phase: 0
mode: probe
backbone:
  kind: encoder
  model_id: google-bert/bert-base-multilingual-cased
  dtype: float32
router:
  hidden_size: 768
  mlp_hidden: 512
  mlp_out: 256
training:
  lr: 3.0e-4
  epochs: 5
  batch_size: 64
  weight_decay: 0.01
  fp16: false
  seed: 42
eval:
  ece_bins: 15
  bootstrap_samples: 1000
  gates:
    macro_f1: 0.70
    hard_negative_accuracy: 0.75
    ece_max: 0.15
```

`configs/phase1_mvp.yaml`:

```yaml
phase: 1
backbone:
  model_id: Qwen/Qwen2.5-Coder-0.5B-Instruct
  dtype: float16
agent:
  regime: NORMAL
  observation_max_tokens: 800
  tools_timeout_s: 60
evaluation:
  n_tasks: 500
  test_subset: 100
  baselines:
    - pure_generation
    - always_answer
    - always_test
    - keyword_router
    - react_loop
    - adaptive_laya
```

`configs/phase2_lora.yaml`:

```yaml
phase: 2
mode: lora
backbone:
  model_id: Qwen/Qwen2.5-Coder-0.5B-Instruct
  dtype: float16
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
router:
  hidden_size: 896
  mlp_hidden: 512
  mlp_out: 256
loss:
  lambda_action: 1.0
  lambda_tool: 1.0
  lambda_terminal: 1.0
  lambda_conf: 0.5
  lambda_budget: 0.0
training:
  lr: 2.0e-4
  micro_batch_size: 1
  gradient_accumulation: 16
  epochs: 1
  fp16: true
  local_steps: 5000
rounds: 2
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: PASS (all).

- [ ] **Step 13: Commit**

```powershell
git add schemas configs tests
git commit -m "feat: schemas and configs for state/action/tool/budget/utility/regimes"
```

---

### Task 2: State serializer (spec §6, §45 item 6)

**Files:**
- Create: `controller/state_serializer.py`
- Test: `tests/test_state_serializer.py`

- [ ] **Step 1: Write failing test**

```python
import json
from pathlib import Path

import yaml

from controller.state_serializer import build_state, serialize_state

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state_serializer.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement `controller/state_serializer.py`**

```python
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
        "user_request": str(record.get("user_request", "")),
        "repository_summary": str(record.get("repository_summary", "")),
        "relevant_file_context": str(record.get("relevant_file_context", "")),
        "recent_changes": str(record.get("recent_changes", "")),
        "previous_actions": list(record.get("previous_actions", [])),
        "test_results": str(record.get("test_results", "")),
        "previous_tool_result": str(record.get("previous_tool_result", "")),
        "remaining_token_budget": int(remaining_tokens),
        "remaining_tool_budget": int(remaining_tools),
        "remaining_latency_budget_ms": int(remaining_latency_ms),
        "step_index": int(record.get("step_index", 0)),
        "previous_confidence": float(record.get("previous_confidence", 0.0)),
    }


def budget_vector(state: dict[str, Any], regime: dict[str, Any]) -> list[float]:
    max_tokens = max(int(regime.get("max_tokens", 1)), 1)
    max_tools = max(int(regime.get("max_tools", 1)), 1)
    max_latency = max(int(regime.get("max_latency_ms", 1)), 1)
    max_steps = max(int(regime.get("max_steps", 1)), 1)
    return [
        min(state["remaining_token_budget"] / max_tokens, 1.0),
        min(state["remaining_tool_budget"] / max_tools, 1.0),
        min(state["remaining_latency_budget_ms"] / max_latency, 1.0),
        min(state["step_index"] / max_steps, 1.0),
    ]


def serialize_state(
    state: dict[str, Any],
    regime: dict[str, Any] | None = None,
    max_tokens: int = 512,
) -> str:
    regime = regime or {"max_tokens": 1, "max_tools": 1, "max_latency_ms": 1, "max_steps": 1}
    b = budget_vector(state, regime)
    b_str = ",".join(f"{x:.2f}" for x in b)
    lines = [
        f"REQ: {state['user_request']}",
        f"REPO: {state['repository_summary']}",
        f"CTX: {state['relevant_file_context']}",
        f"CHANGES: {state['recent_changes']}",
        f"PREV_ACTIONS: {' '.join(state['previous_actions']) or 'none'}",
        f"TESTS: {state['test_results'] or 'none'}",
        f"PREV_OBS: {state['previous_tool_result'] or 'none'}",
        (
            f"step={state['step_index']} tok={state['remaining_token_budget']} "
            f"tools={state['remaining_tool_budget']} lat_ms={state['remaining_latency_budget_ms']} "
            f"conf={state['previous_confidence']:.2f} b=[{b_str}]"
        ),
    ]
    text = "\n".join(lines)
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])
    return text


def state_to_json(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state_serializer.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add controller/state_serializer.py tests/test_state_serializer.py
git commit -m "feat: state serializer with normalized budget vector"
```

---

### Task 3: Cost ledger (spec §13, §45 item 7)

**Files:**
- Create: `controller/cost_ledger.py`
- Test: `tests/test_cost_ledger.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path

from controller.cost_ledger import CostLedger, load_cost_coeffs

ROOT = Path(__file__).resolve().parents[1]


def test_total_formula():
    coeffs = {"alpha_tool_calls": 0.05, "beta_router_forwards": 0.001, "gamma_wall_ms": 1e-5}
    led = CostLedger(coeffs)
    led.add(input_tokens=100, output_tokens=50, tool_calls=2, router_forwards=3, wall_ms=1000)
    expected = 150 + 0.05 * 2 + 0.001 * 3 + 1e-5 * 1000
    assert abs(led.total() - expected) < 1e-9


def test_accumulates():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    led.add(input_tokens=10)
    led.add(output_tokens=5)
    assert led.total() == 15
    d = led.as_dict()
    assert d["input_tokens"] == 10 and d["output_tokens"] == 5


def test_load_from_config():
    coeffs = load_cost_coeffs(ROOT / "configs" / "utility_config.yaml")
    assert "alpha_tool_calls" in coeffs


def test_utility():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    led.add(output_tokens=1000)
    u = led.utility(success=True, cfg={"success_reward": 1.0, "alpha_tokens": 2e-4,
                                       "beta_tool_calls": 0.05, "gamma_latency_ms": 1e-5,
                                       "delta_router_forwards": 1e-3})
    assert abs(u - (1.0 - 1000 * 2e-4)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cost_ledger.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `controller/cost_ledger.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_cost_coeffs(path: str | Path) -> dict[str, float]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in cfg["cost_ledger"].items()}


class CostLedger:
    def __init__(self, coeffs: dict[str, float]):
        self.coeffs = coeffs
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = 0
        self.router_forwards = 0
        self.wall_ms = 0.0

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
        router_forwards: int = 0,
        wall_ms: float = 0.0,
    ) -> None:
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.tool_calls += int(tool_calls)
        self.router_forwards += int(router_forwards)
        self.wall_ms += float(wall_ms)

    def total(self) -> float:
        tokens = self.input_tokens + self.output_tokens
        return (
            tokens
            + self.coeffs["alpha_tool_calls"] * self.tool_calls
            + self.coeffs["beta_router_forwards"] * self.router_forwards
            + self.coeffs["gamma_wall_ms"] * self.wall_ms
        )

    def utility(self, success: bool, cfg: dict[str, float]) -> float:
        reward = float(cfg["success_reward"]) if success else 0.0
        return (
            reward
            - float(cfg["alpha_tokens"]) * (self.input_tokens + self.output_tokens)
            - float(cfg["beta_tool_calls"]) * self.tool_calls
            - float(cfg["gamma_latency_ms"]) * self.wall_ms
            - float(cfg["delta_router_forwards"]) * self.router_forwards
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "router_forwards": self.router_forwards,
            "wall_ms": self.wall_ms,
            "total_cost": self.total(),
        }

    def reset(self) -> None:
        self.__init__(self.coeffs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cost_ledger.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add controller/cost_ledger.py tests/test_cost_ledger.py
git commit -m "feat: cost ledger with total cost and utility"
```

---

### Task 4: Budget manager and loop guard (spec §7, §12)

**Files:**
- Create: `controller/budget_manager.py`, `controller/loop_guard.py`
- Test: `tests/test_budget_guard.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path

from controller.budget_manager import BudgetManager, load_regime
from controller.loop_guard import LoopGuard

ROOT = Path(__file__).resolve().parents[1]


def test_regime_loading():
    regime = load_regime(ROOT / "configs" / "regimes.yaml", "LOW")
    assert regime["max_tokens"] == 250 and regime["max_tools"] == 1


def test_budget_vector_bounds():
    mgr = BudgetManager(load_regime(ROOT / "configs" / "regimes.yaml", "NORMAL"))
    v = mgr.vector()
    assert len(v) == 4
    assert all(0.0 <= x <= 1.0 for x in v)


def test_budget_spending():
    mgr = BudgetManager(load_regime(ROOT / "configs" / "regimes.yaml", "LOW"))
    assert mgr.can_spend_tokens(200)
    mgr.record_tokens(200)
    assert not mgr.can_spend_tokens(100)
    assert mgr.can_call_tool()
    mgr.record_tool()
    assert not mgr.can_call_tool()


def test_loop_guard_blocks_over_limit():
    guard = LoopGuard({"max_steps": 2, "max_tool_calls": 1,
                       "max_generated_tokens": 10, "max_observation_tokens": 10,
                       "max_wall_time_ms": 1000})
    assert guard.check(step_index=0, generated_tokens=0, tool_calls=0, obs_tokens=0, wall_ms=0)
    assert not guard.check(step_index=2, generated_tokens=0, tool_calls=0, obs_tokens=0, wall_ms=0)
    assert guard.reason() == "max_steps"


def test_loop_guard_blocks_tool_calls():
    guard = LoopGuard({"max_steps": 8, "max_tool_calls": 1,
                       "max_generated_tokens": 10, "max_observation_tokens": 10,
                       "max_wall_time_ms": 1000})
    assert not guard.check(step_index=0, generated_tokens=0, tool_calls=2, obs_tokens=0, wall_ms=0)
    assert guard.reason() == "max_tool_calls"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_budget_guard.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `controller/budget_manager.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_regime(path: str | Path, name: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(cfg["regimes"][name])


class BudgetManager:
    def __init__(self, regime: dict[str, Any]):
        self.regime = regime
        self.max_tokens = int(regime["max_tokens"])
        self.max_tools = int(regime["max_tools"])
        self.max_latency_ms = int(regime["max_latency_ms"])
        self.max_steps = int(regime["max_steps"])
        self.used_tokens = 0
        self.used_tools = 0
        self.used_latency_ms = 0
        self.step = 0

    @property
    def remaining_tokens(self) -> int:
        return max(self.max_tokens - self.used_tokens, 0)

    @property
    def remaining_tools(self) -> int:
        return max(self.max_tools - self.used_tools, 0)

    @property
    def remaining_latency_ms(self) -> int:
        return max(self.max_latency_ms - self.used_latency_ms, 0)

    def vector(self) -> list[float]:
        return [
            self.remaining_tokens / max(self.max_tokens, 1),
            self.remaining_tools / max(self.max_tools, 1),
            self.remaining_latency_ms / max(self.max_latency_ms, 1),
            min(self.step / max(self.max_steps, 1), 1.0),
        ]

    def can_spend_tokens(self, n: int) -> bool:
        return self.used_tokens + n <= self.max_tokens

    def can_call_tool(self) -> bool:
        return self.used_tools < self.max_tools

    def record_tokens(self, n: int) -> None:
        self.used_tokens += n

    def record_tool(self) -> None:
        self.used_tools += 1

    def record_latency(self, ms: float) -> None:
        self.used_latency_ms += ms

    def record_step(self) -> None:
        self.step += 1

    def exhausted(self) -> bool:
        return (
            self.used_tokens >= self.max_tokens
            or self.used_tools >= self.max_tools
            or self.used_latency_ms >= self.max_latency_ms
            or self.step >= self.max_steps
        )
```

- [ ] **Step 4: Implement `controller/loop_guard.py`**

```python
from __future__ import annotations

from typing import Any


class LoopGuard:
    def __init__(self, limits: dict[str, Any]):
        self.limits = {
            "max_steps": int(limits["max_steps"]),
            "max_tool_calls": int(limits["max_tool_calls"]),
            "max_generated_tokens": int(limits["max_generated_tokens"]),
            "max_observation_tokens": int(limits["max_observation_tokens"]),
            "max_wall_time_ms": int(limits["max_wall_time_ms"]),
        }
        self._reason: str | None = None

    def reason(self) -> str | None:
        return self._reason

    def check(
        self,
        step_index: int,
        generated_tokens: int,
        tool_calls: int,
        obs_tokens: int,
        wall_ms: float,
    ) -> bool:
        checks = [
            ("max_steps", step_index < self.limits["max_steps"]),
            ("max_tool_calls", tool_calls < self.limits["max_tool_calls"]),
            ("max_generated_tokens", generated_tokens < self.limits["max_generated_tokens"]),
            ("max_observation_tokens", obs_tokens < self.limits["max_observation_tokens"]),
            ("max_wall_time_ms", wall_ms < self.limits["max_wall_time_ms"]),
        ]
        for name, ok in checks:
            if not ok:
                self._reason = name
                return False
        self._reason = None
        return True
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_budget_guard.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```powershell
git add controller/budget_manager.py controller/loop_guard.py tests/test_budget_guard.py
git commit -m "feat: budget manager and loop guard"
```

---

### Task 5: Router model (spec §5)

**Files:**
- Create: `model/routing_heads.py`, `model/router.py`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write failing test**

```python
import torch

from model.router import ACTIONS, TERMINALS, TOOLS, AdaptiveRouter


def test_output_shapes():
    router = AdaptiveRouter(hidden_size=64, mlp_hidden=32, mlp_out=16)
    h = torch.randn(4, 64)
    budget = torch.rand(4, 4)
    out = router(h, budget)
    assert out.action_logits.shape == (4, len(ACTIONS))
    assert out.tool_logits.shape == (4, len(TOOLS))
    assert out.terminal_logits.shape == (4, len(TERMINALS))
    assert out.confidence.shape == (4,)
    assert torch.all((out.confidence >= 0) & (out.confidence <= 1))


def test_param_count_1m_to_5m():
    router = AdaptiveRouter(hidden_size=896, mlp_hidden=512, mlp_out=256)
    n = sum(p.numel() for p in router.parameters())
    assert 1_000_000 <= n <= 5_000_000


def test_predict_action_masks_tool_when_not_tool():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    decision = router.predict(torch.randn(1, 32), torch.rand(1, 4))
    assert decision["action"] in ACTIONS
    if decision["action"] != "TOOL":
        assert decision["tool"] is None
    assert 0.0 <= decision["confidence"] <= 1.0
    assert decision["terminal"] in TERMINALS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_router.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `model/routing_heads.py`**

```python
from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConfidenceHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x).squeeze(-1))
```

- [ ] **Step 4: Implement `model/router.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.routing_heads import ConfidenceHead, Head

ACTIONS = ["ANSWER", "TOOL", "ASK"]
TOOLS = ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]
TERMINALS = ["CONTINUE", "DONE", "ESCALATE"]


@dataclass
class RouterOutput:
    action_logits: torch.Tensor
    tool_logits: torch.Tensor
    terminal_logits: torch.Tensor
    confidence: torch.Tensor


class AdaptiveRouter(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 512, mlp_out: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_out),
        )
        self.budget_proj = nn.Linear(4, mlp_out)
        self.action_head = Head(mlp_out, len(ACTIONS))
        self.tool_head = Head(mlp_out, len(TOOLS))
        self.terminal_head = Head(mlp_out, len(TERMINALS))
        self.confidence_head = ConfidenceHead(mlp_out)

    def forward(self, h: torch.Tensor, budget: torch.Tensor | None = None) -> RouterOutput:
        x = self.mlp(self.norm(h))
        if budget is not None:
            x = x + self.budget_proj(budget)
        return RouterOutput(
            action_logits=self.action_head(x),
            tool_logits=self.tool_head(x),
            terminal_logits=self.terminal_head(x),
            confidence=self.confidence_head(x),
        )

    @torch.no_grad()
    def predict(self, h: torch.Tensor, budget: torch.Tensor | None = None) -> dict:
        out = self.forward(h, budget)
        action_idx = int(out.action_logits.argmax(-1).item())
        action = ACTIONS[action_idx]
        tool = None
        if action == "TOOL":
            tool = TOOLS[int(out.tool_logits.argmax(-1).item())]
        return {
            "action": action,
            "tool": tool,
            "terminal": TERMINALS[int(out.terminal_logits.argmax(-1).item())],
            "confidence": float(out.confidence.item()),
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_router.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```powershell
git add model/routing_heads.py model/router.py tests/test_router.py
git commit -m "feat: adaptive router with action/tool/confidence/terminal heads"
```

---

### Task 6: Losses and calibration (spec §14, §18, §27, §36)

**Files:**
- Create: `model/losses.py`, `model/calibration.py`
- Test: `tests/test_losses_calibration.py`

- [ ] **Step 1: Write failing test**

```python
import torch

from model.calibration import apply_temperature, expected_calibration_error, fit_temperature
from model.losses import router_loss
from model.router import AdaptiveRouter


def _coeffs():
    return {"lambda_action": 1.0, "lambda_tool": 1.0,
            "lambda_terminal": 1.0, "lambda_conf": 0.5, "lambda_budget": 0.0}


def test_router_loss_scalar_and_backward():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=32, mlp_hidden=16, mlp_out=8)
    out = router(torch.randn(6, 32), torch.rand(6, 4))
    labels = {
        "action": torch.tensor([0, 1, 2, 1, 0, 1]),
        "tool": torch.tensor([-1, 0, -1, 2, -1, -1]),
        "terminal": torch.tensor([0, 0, 1, 0, 2, 0]),
        "confidence": torch.rand(6),
    }
    parts = router_loss(out, labels, _coeffs())
    assert parts["loss"].requires_grad
    assert "action_loss" in parts and "tool_loss" in parts
    parts["loss"].backward()


def test_tool_loss_masked_when_not_tool():
    torch.manual_seed(0)
    router = AdaptiveRouter(hidden_size=16, mlp_hidden=8, mlp_out=8)
    out = router(torch.randn(4, 16), torch.rand(4, 4))
    labels = {
        "action": torch.zeros(4, dtype=torch.long),
        "tool": torch.full((4,), -1, dtype=torch.long),
        "terminal": torch.zeros(4, dtype=torch.long),
        "confidence": torch.rand(4),
    }
    parts = router_loss(out, labels, _coeffs())
    assert float(parts["tool_loss"]) == 0.0


def test_ece_bounds():
    conf = torch.tensor([0.9, 0.9, 0.1, 0.1])
    correct = torch.tensor([1, 1, 0, 0])
    ece = expected_calibration_error(conf, correct, n_bins=2)
    assert 0.0 <= ece <= 1.0


def test_temperature_scaling():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    labels = torch.tensor([0, 1])
    t = fit_temperature(logits, labels, lr=0.1, steps=50)
    assert t > 0
    scaled = apply_temperature(logits, t)
    assert scaled.shape == logits.shape
    assert torch.allclose(scaled, logits / t, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_losses_calibration.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `model/losses.py`**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.router import ACTIONS, RouterOutput


def router_loss(
    out: RouterOutput,
    labels: dict[str, torch.Tensor],
    coeffs: dict[str, float],
) -> dict[str, torch.Tensor]:
    action = labels["action"]
    tool = labels["tool"]
    terminal = labels["terminal"]
    confidence = labels["confidence"]

    action_loss = F.cross_entropy(out.action_logits, action)
    tool_mask = action == ACTIONS.index("TOOL")
    if tool_mask.any():
        tool_loss = F.cross_entropy(out.tool_logits[tool_mask], tool[tool_mask])
    else:
        tool_loss = out.action_logits.sum() * 0.0
    terminal_loss = F.cross_entropy(out.terminal_logits, terminal)
    conf_loss = F.mse_loss(out.confidence, confidence)

    loss = (
        float(coeffs.get("lambda_action", 1.0)) * action_loss
        + float(coeffs.get("lambda_tool", 1.0)) * tool_loss
        + float(coeffs.get("lambda_terminal", 1.0)) * terminal_loss
        + float(coeffs.get("lambda_conf", 1.0)) * conf_loss
    )
    return {
        "loss": loss,
        "action_loss": action_loss.detach(),
        "tool_loss": tool_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "conf_loss": conf_loss.detach(),
    }


def utility_weighted_action_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cost_weights: torch.Tensor,
) -> torch.Tensor:
    per = F.cross_entropy(logits, targets, reduction="none")
    return (per * cost_weights).mean()
```

- [ ] **Step 4: Implement `model/calibration.py`**

```python
from __future__ import annotations

import torch


def expected_calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int = 15,
) -> float:
    confidence = confidence.detach().float().cpu()
    correct = correct.detach().float().cpu()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = confidence.numel()
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if mask.any():
            acc = correct[mask].mean().item()
            conf = confidence[mask].mean().item()
            ece += (mask.sum().item() / n) * abs(acc - conf)
    return float(ece)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lr: float = 0.01,
    steps: int = 200,
) -> float:
    temp = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([temp], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(temp.detach().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-6)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_losses_calibration.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```powershell
git add model/losses.py model/calibration.py tests/test_losses_calibration.py
git commit -m "feat: masked multi-head router loss and calibration utilities"
```

---

### Task 7: Frozen backbone feature extraction (spec §2, §3, §18)

**Files:**
- Create: `model/backbone.py`
- Test: `tests/test_backbone.py`

- [ ] **Step 1: Write failing test** (offline: tiny random Qwen2 config + cached tokenizer)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from model.backbone import ROUTE_TOKEN, FrozenDecoderBackbone


def tiny_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def test_route_hidden_shape():
    model, tok = tiny_model_and_tokenizer()
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    h = bb.route_hidden("def add(a, b): return a + b")
    assert h.shape[-1] == 32
    assert h.ndim == 2


def test_route_hidden_deterministic_and_frozen():
    model, tok = tiny_model_and_tokenizer()
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    h1 = bb.route_hidden("x = 1")
    h2 = bb.route_hidden("x = 1")
    assert torch.allclose(h1, h2, atol=1e-5)
    assert all(not p.requires_grad for p in model.parameters())


def test_route_token_appended():
    model, tok = tiny_model_and_tokenizer()
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    ids = tok.encode("hi", add_special_tokens=False)
    appended = bb._append_route(ids)
    route_ids = tok.encode(ROUTE_TOKEN, add_special_tokens=False)
    assert appended[-1] in set(route_ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backbone.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `model/backbone.py`**

```python
from __future__ import annotations

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROUTE_TOKEN = "<|route|>"


def load_decoder(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def load_encoder(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


class FrozenDecoderBackbone:
    def __init__(self, model, tokenizer, route_token: str = ROUTE_TOKEN, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.route_token = route_token
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _append_route(self, ids: list[int]) -> list[int]:
        return ids + self.tokenizer.encode(self.route_token, add_special_tokens=False)

    @torch.no_grad()
    def route_hidden(self, text: str, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
        ids = self._append_route(ids)
        input_ids = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states[-1][:, -1, :]


class FrozenEncoderBackbone:
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def route_hidden(self, text: str, max_length: int = 512) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backbone.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add model/backbone.py tests/test_backbone.py
git commit -m "feat: frozen decoder/encoder backbones with route-token hidden state"
```

---

### Task 8: Seed decision dataset generator (spec §15–§17, §45 items 8–10)

**Files:**
- Create: `scripts/data_generation/generate_decisions.py`, `scripts/data_generation/hard_negatives.py`, `scripts/data_generation/counterfactuals.py`, `scripts/data_generation/perturb.py`
- Test: `tests/test_data_generation.py`

- [ ] **Step 1: Write failing test**

```python
import json

from scripts.data_generation.counterfactuals import label_counterfactuals
from scripts.data_generation.generate_decisions import generate_dataset, write_dataset
from scripts.data_generation.hard_negatives import add_hard_negatives
from scripts.data_generation.perturb import perturb_numbers, rename_identifiers

REGIMES = {
    "LOW": {"success_reward": 1.0, "alpha_tokens": 2e-4, "beta_tool_calls": 5e-2, "gamma_latency_ms": 1e-5},
    "NORMAL": {"success_reward": 1.0, "alpha_tokens": 1e-4, "beta_tool_calls": 2e-2, "gamma_latency_ms": 5e-6},
    "HIGH": {"success_reward": 1.0, "alpha_tokens": 5e-5, "beta_tool_calls": 1e-2, "gamma_latency_ms": 2e-6},
}


def test_generate_balanced_ids_unique():
    data = generate_dataset(n=300, seed=0)
    assert len(data) == 300
    ids = [d["id"] for d in data]
    assert len(set(ids)) == 300
    assert {d["label_action"] for d in data} == {"ANSWER", "TOOL", "ASK"}


def test_write_and_reload(tmp_path):
    data = generate_dataset(n=50, seed=1)
    out = write_dataset(data, tmp_path)
    assert out.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["count"] == 50
    assert manifest["sha256"]
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == 50


def test_hard_negatives_flagged():
    data = generate_dataset(n=200, seed=2)
    hn = add_hard_negatives(data, n=40, seed=3)
    flagged = [d for d in hn if d["hard_negative"]]
    assert len(flagged) >= 40
    assert all(d["label_action"] in {"ANSWER", "TOOL", "ASK"} for d in hn)


def test_counterfactuals_multi_regime_labels():
    data = generate_dataset(n=60, seed=4)
    labeled = label_counterfactuals(data, REGIMES)
    assert set(labeled[0]["regime_best_action"]) == {"LOW", "NORMAL", "HIGH"}
    for d in labeled:
        for v in d["regime_best_action"].values():
            assert v in {"ANSWER", "TOOL", "ASK"}


def test_perturb_helpers():
    original = "timeout = 30 seconds, limit = 100"
    s = perturb_numbers(original, seed=1)
    assert s != original
    s2 = rename_identifiers("def compute_total(items):")
    assert "compute_total" not in s2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_generation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/data_generation/generate_decisions.py`**

```python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

TASK_FAMILIES = ["fix_bug", "write_func", "explain_code", "refactor", "add_test", "debug_failing_test"]

ANSWER_PROMPTS = [
    "What does this function do?",
    "Explain the time complexity of this loop.",
    "Summarize this repository structure.",
    "What is wrong with this docstring?",
]

TOOL_PROMPTS = [
    "Fix the failing test in {file}.",
    "Make the linter pass on {file}.",
    "Find where {symbol} is defined.",
    "Run the test suite and fix the failure.",
]

ASK_PROMPTS = [
    "Make the function better.",
    "Fix the bug.",
    "Improve performance.",
    "Clean this up.",
]


def _default_branches(action: str) -> dict[str, dict[str, float]]:
    branches = {
        "ANSWER": {"success": 0.55, "tokens": 120, "tools": 0, "latency_ms": 400},
        "TOOL": {"success": 0.85, "tokens": 60, "tools": 1, "latency_ms": 1500},
        "ASK": {"success": 0.70, "tokens": 40, "tools": 0, "latency_ms": 300},
    }
    branches["_chosen"] = {"action": action}
    return branches


def _base_record(i: int, action: str, prompt: str, family: str, difficulty: str, rng: random.Random) -> dict[str, Any]:
    tool = None
    terminal = "CONTINUE"
    if action == "TOOL":
        tool = rng.choice(["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"])
    if action == "ANSWER" and rng.random() < 0.5:
        terminal = "DONE"
    return {
        "id": f"dec-{i:06d}",
        "prompt": prompt,
        "user_request": prompt,
        "repository_summary": "Small Python repository with pytest suite.",
        "relevant_file_context": "def helper(x):\n    return x",
        "recent_changes": "none",
        "previous_actions": [],
        "test_results": "",
        "previous_tool_result": "",
        "step_index": 0,
        "previous_confidence": 0.0,
        "label_action": action,
        "label_tool": tool,
        "label_terminal": terminal,
        "label_confidence": round(rng.uniform(0.55, 0.95), 3),
        "task_family": family,
        "difficulty": difficulty,
        "repository": f"example/repo-{i % 7}",
        "hard_negative": False,
        "branches": _default_branches(action),
    }


def generate_dataset(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    actions = ["ANSWER", "TOOL", "ASK"]
    for i in range(n):
        action = actions[i % 3]
        family = TASK_FAMILIES[i % len(TASK_FAMILIES)]
        difficulty = ["easy", "medium", "hard"][i % 3]
        if action == "ANSWER":
            prompt = ANSWER_PROMPTS[i % len(ANSWER_PROMPTS)]
        elif action == "TOOL":
            prompt = TOOL_PROMPTS[i % len(TOOL_PROMPTS)].format(
                file=f"module_{i % 7}.py", symbol=f"func_{i % 11}"
            )
        else:
            prompt = ASK_PROMPTS[i % len(ASK_PROMPTS)]
        records.append(_base_record(i, action, prompt, family, difficulty, rng))
    rng.shuffle(records)
    return records


def write_dataset(records: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "decisions.jsonl"
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    jsonl_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "count": len(records),
        "sha256": digest,
        "source": "synthetic-seed-generator",
        "version": "v1",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return jsonl_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/processed/decisions_v1")
    args = ap.parse_args()
    records = generate_dataset(args.n, args.seed)
    path = write_dataset(records, args.out)
    print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `scripts/data_generation/hard_negatives.py`**

```python
from __future__ import annotations

import copy
import random
from typing import Any

HARD_TEMPLATES = [
    {
        "prompt": "Run the tests and fix the failure.",
        "label_action": "ANSWER",
        "note": "sounds like TOOL but missing repo context, answer with clarifying plan",
    },
    {
        "prompt": "What does this function return?",
        "label_action": "TOOL",
        "note": "question-like but requires reading context to answer precisely",
    },
    {
        "prompt": "Fix everything that is broken.",
        "label_action": "ASK",
        "note": "underspecified scope requires clarification",
    },
    {
        "prompt": "Add type hints across the package.",
        "label_action": "ASK",
        "note": "ambiguous target package",
    },
]


def add_hard_negatives(records: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = [copy.deepcopy(r) for r in records]
    start = len(out)
    for j in range(n):
        tmpl = HARD_TEMPLATES[j % len(HARD_TEMPLATES)]
        rec = copy.deepcopy(out[j % max(len(out), 1)])
        rec["id"] = f"hn-{start + j:06d}"
        rec["prompt"] = tmpl["prompt"]
        rec["user_request"] = tmpl["prompt"]
        rec["label_action"] = tmpl["label_action"]
        rec["label_tool"] = None if tmpl["label_action"] != "TOOL" else "READ_CONTEXT"
        rec["hard_negative"] = True
        rec["note"] = tmpl["note"]
        rec["label_confidence"] = round(rng.uniform(0.5, 0.85), 3)
        out.append(rec)
    rng.shuffle(out)
    return out
```

- [ ] **Step 5: Implement `scripts/data_generation/counterfactuals.py`**

```python
from __future__ import annotations

import copy
from typing import Any


def branch_utility(branch: dict[str, float], regime_coeffs: dict[str, float]) -> float:
    return (
        float(regime_coeffs.get("success_reward", 1.0)) * branch["success"]
        - regime_coeffs["alpha_tokens"] * branch["tokens"]
        - regime_coeffs["beta_tool_calls"] * branch["tools"]
        - regime_coeffs["gamma_latency_ms"] * branch["latency_ms"]
    )


def label_counterfactuals(
    records: list[dict[str, Any]],
    regimes: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        rec = copy.deepcopy(rec)
        branches = rec.get("branches", {})
        best = {}
        for regime_name, coeffs in regimes.items():
            scored = {
                action: branch_utility(branch, coeffs)
                for action, branch in branches.items()
                if action in {"ANSWER", "TOOL", "ASK"}
            }
            best[regime_name] = max(scored, key=scored.get)
        rec["regime_best_action"] = best
        out.append(rec)
    return out
```

- [ ] **Step 6: Implement `scripts/data_generation/perturb.py`**

```python
from __future__ import annotations

import random
import re


def perturb_numbers(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    return re.sub(
        r"\b(\d+)\b",
        lambda m: str(max(1, int(m.group(1)) + rng.randint(-5, 5))),
        text,
    )


def rename_identifiers(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
    mapping = {}
    for name in set(names):
        if name in {"def", "return", "import", "class", "self"}:
            continue
        mapping[name] = f"ref_{rng.randint(100, 999)}"
    for old, new in mapping.items():
        text = text.replace(old, new)
    return text


def paraphrase_prompt(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    prefixes = ["Please ", "Kindly ", "Can you ", ""]
    suffixes = ["", " Thanks.", " Be specific."]
    return rng.choice(prefixes) + text.strip().rstrip(".") + rng.choice(suffixes)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_generation.py -v`
Expected: PASS (all).

- [ ] **Step 8: Generate the real seed dataset (10k + 1.5k hard negatives + counterfactual labels)**

Run:

```powershell
python scripts/data_generation/generate_decisions.py --n 10000 --seed 0 --out data/processed/decisions_v1
python -c "import json,yaml; from pathlib import Path; from scripts.data_generation.hard_negatives import add_hard_negatives; from scripts.data_generation.counterfactuals import label_counterfactuals; from scripts.data_generation.generate_decisions import write_dataset; recs=[json.loads(l) for l in Path('data/processed/decisions_v1/decisions.jsonl').read_text().splitlines()]; recs=add_hard_negatives(recs, n=1500, seed=1); util=yaml.safe_load(Path('configs/utility_config.yaml').read_text())['utility']; regimes={k: dict(util) for k in ['LOW','NORMAL','HIGH']}; regimes['LOW']['alpha_tokens']=2e-4; regimes['LOW']['beta_tool_calls']=5e-2; regimes['NORMAL']['alpha_tokens']=1e-4; regimes['NORMAL']['beta_tool_calls']=2e-2; regimes['HIGH']['alpha_tokens']=5e-5; regimes['HIGH']['beta_tool_calls']=1e-2; recs=label_counterfactuals(recs, regimes); write_dataset(recs, 'data/decisions/decisions_v1'); print(len(recs))"
```

Expected: `wrote 10000 records...` then `11500`.

- [ ] **Step 9: Run full Part 1 suite and commit**

Run: `python -m pytest tests/ -v`
Expected: all PASS.

```powershell
git add -A
git commit -m "feat: seed decision generator, hard negatives, counterfactual labeler, perturbation"
```

---

**End of Part 1.** Next: `2026-09-24-adaptive-laya-part2-probe-agent.md` (Tasks 9–15).
