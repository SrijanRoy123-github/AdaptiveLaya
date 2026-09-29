from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score_run(run: dict[str, Any], reference_answer: str) -> dict[str, Any]:
    final = str(run.get("final_answer", "")).strip().lower()
    ref = reference_answer.strip().lower()
    success = bool(run.get("finished")) and final == ref and run.get("stop_reason") == "done"
    cost = run.get("cost", {})
    return {
        "success": success,
        "total_cost": float(cost.get("total_cost", 0.0)),
        "input_tokens": int(cost.get("input_tokens", 0)),
        "output_tokens": int(cost.get("output_tokens", 0)),
        "tool_calls": int(cost.get("tool_calls", 0)),
        "router_forwards": int(cost.get("router_forwards", 0)),
        "wall_ms": float(cost.get("wall_ms", 0.0)),
        "steps": int(run.get("steps", 0)),
        "stop_reason": run.get("stop_reason"),
    }


def aggregate(scored_runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(scored_runs), 1)
    return {
        "n": len(scored_runs),
        "success_rate": sum(r["success"] for r in scored_runs) / n,
        "mean_cost": sum(r["total_cost"] for r in scored_runs) / n,
        "mean_tool_calls": sum(r["tool_calls"] for r in scored_runs) / n,
        "mean_router_forwards": sum(r["router_forwards"] for r in scored_runs) / n,
        "mean_wall_ms": sum(r["wall_ms"] for r in scored_runs) / n,
        "mean_steps": sum(r["steps"] for r in scored_runs) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, required=True, help="JSONL of run results with reference answers")
    ap.add_argument("--out", type=str, default="results/agent_eval.json")
    args = ap.parse_args()

    per_system: dict[str, list[dict[str, Any]]] = {}
    for line in Path(args.runs).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        system = row["system"]
        scored = score_run(row["run"], row["reference_answer"])
        per_system.setdefault(system, []).append(scored)

    summary = {name: aggregate(runs) for name, runs in per_system.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
