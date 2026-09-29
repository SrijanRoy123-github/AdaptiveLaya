from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

_frontier_points: list[dict[str, Any]] = []


def frontier_point(success_rate: float, total_cost: float) -> dict[str, Any]:
    dominated = any(
        p["success_rate"] >= success_rate and p["mean_cost"] <= total_cost
        for p in _frontier_points
    )
    point: dict[str, Any] = {
        "success_rate": success_rate,
        "mean_cost": total_cost,
        "pareto": not dominated,
    }
    _frontier_points.append(point)
    return point


def mark_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: (r["mean_cost"], -r["success_rate"]))
    best_success = -1.0
    for row in sorted_rows:
        if row["success_rate"] > best_success:
            row["pareto"] = True
            best_success = row["success_rate"]
        else:
            row["pareto"] = False
    return rows


def write_frontier_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "success_rate", "mean_cost", "pareto"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(row.get(k)).lower() if k == "pareto" else row.get(k) for k in ["system", "success_rate", "mean_cost", "pareto"]})
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True, help="agent_eval.json path")
    ap.add_argument("--out", type=str, default="results/frontier.csv")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = [
        {"system": name, "success_rate": m["success_rate"], "mean_cost": m["mean_cost"]}
        for name, m in summary.items()
    ]
    rows = mark_pareto(rows)
    path = write_frontier_csv(rows, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
