from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

TASK_FAMILIES = ["fix_bug", "write_func", "explain_code", "refactor", "add_test", "debug_failing_test",
                 "vlsi_design", "cad_layout", "math_proof", "reasoning", "book_summary", "science_qa"]

ANSWER_PROMPTS = [
    "What does this function do?",
    "Explain the time complexity of this loop.",
    "Summarize this repository structure.",
    "What is wrong with this docstring?",
    "Explain this VLSI circuit block.",
    "Describe this CAD layout.",
    "Solve this math problem.",
    "Reason through this logic puzzle.",
    "Summarize this book chapter.",
    "Explain this science concept.",
]

TOOL_PROMPTS = [
    "Fix the failing test in {file}.",
    "Make the linter pass on {file}.",
    "Find where {symbol} is defined.",
    "Run the test suite and fix the failure.",
    "Simulate this SPICE netlist and fix the failure.",
    "Run DRC on this layout and fix the violations.",
    "Compute the result of this equation.",
    "Deduce the next step in this proof.",
    "Summarize this book excerpt.",
    "Query the science dataset for this concept.",
]

ASK_PROMPTS = [
    "Make the function better.",
    "Fix the bug.",
    "Improve performance.",
    "Clean this up.",
    "Improve this VLSI design.",
    "Optimize this CAD floorplan.",
    "Explain this math derivation.",
    "Clarify this reasoning step.",
    "Improve this book summary.",
    "Deepen this science explanation.",
]


ACTIONS = ["ANSWER", "TOOL", "ASK"]

_BASE_COST_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "ANSWER": {"tokens": (90.0, 160.0), "tools": (0.0, 0.0), "latency_ms": (300.0, 600.0)},
    "TOOL": {"tokens": (40.0, 90.0), "tools": (1.0, 1.0), "latency_ms": (1000.0, 2000.0)},
    "ASK": {"tokens": (20.0, 50.0), "tools": (0.0, 0.0), "latency_ms": (150.0, 450.0)},
}


def _jittered_branch(action: str, success: float, rng: random.Random) -> dict[str, float]:
    lo_t, hi_t = _BASE_COST_RANGES[action]["tokens"]
    lo_tools, hi_tools = _BASE_COST_RANGES[action]["tools"]
    lo_lat, hi_lat = _BASE_COST_RANGES[action]["latency_ms"]
    return {
        "success": success,
        "tokens": rng.uniform(lo_t, hi_t),
        "tools": rng.uniform(lo_tools, hi_tools),
        "latency_ms": rng.uniform(lo_lat, hi_lat),
    }


def build_branches(action: str, rng: random.Random) -> dict[str, dict[str, Any]]:
    """Per-record, action-aware branch stats.

    The chosen ``action`` is constructed to have strictly the best utility
    under the NORMAL regime (success gap >= 0.20 vs NORMAL cost spread
    <= ~0.17). A fraction of records ("swing" records) inflate the chosen
    action's costs so the LOW regime (strict cost penalties) flips its
    argmax to a cheap rival while HIGH (lenient penalties) keeps the
    chosen action — producing genuine counterfactual variation.
    """
    swing = rng.random() < 0.45
    swing_rival = rng.choice([a for a in ACTIONS if a != action]) if swing else None
    branches: dict[str, dict[str, Any]] = {}
    for name in ACTIONS:
        if name == action and swing:
            branches[name] = {
                "success": rng.uniform(0.80, 0.82),
                "tokens": rng.uniform(500.0, 650.0),
                "tools": rng.uniform(2.5, 3.0),
                "latency_ms": rng.uniform(5000.0, 8000.0),
            }
        elif name == action:
            branches[name] = _jittered_branch(name, rng.uniform(0.80, 0.95), rng)
        elif name == swing_rival:
            branches[name] = {
                "success": rng.uniform(0.59, 0.60),
                "tokens": rng.uniform(5.0, 30.0),
                "tools": 0.0,
                "latency_ms": rng.uniform(50.0, 250.0),
            }
        else:
            branches[name] = _jittered_branch(name, rng.uniform(0.30, 0.60), rng)
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
        "branches": build_branches(action, rng),
    }


def generate_dataset(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    actions = ["ANSWER", "TOOL", "ASK"]
    for i in range(n):
        action = actions[i % 3]
        family = rng.choice(TASK_FAMILIES)
        difficulty = rng.choice(["easy", "medium", "hard"])
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
    payload = "\n".join(json.dumps(r, sort_keys=True, allow_nan=False) for r in records) + "\n"
    jsonl_path.write_bytes(payload.encode("utf-8"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "count": len(records),
        "sha256": digest,
        "source": "synthetic-seed-generator",
        "version": "v1",
    }
    (out / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
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
