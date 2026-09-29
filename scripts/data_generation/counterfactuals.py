from __future__ import annotations

import copy
from typing import Any

ACTIONS = ("ANSWER", "TOOL", "ASK")


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
    if not regimes:
        raise ValueError("regimes must be a non-empty mapping")
    out = []
    for rec in records:
        rec = copy.deepcopy(rec)
        branches = rec.get("branches")
        if not isinstance(branches, dict):
            raise TypeError(f"record {rec.get('id', '<unknown>')!r} is missing a 'branches' dict")
        scoreable = [action for action in ACTIONS if action in branches]
        if not scoreable:
            raise ValueError(
                f"record {rec.get('id', '<unknown>')!r} has no ANSWER/TOOL/ASK branches to score"
            )
        best = {}
        for regime_name, coeffs in regimes.items():
            scored = {action: branch_utility(branches[action], coeffs) for action in scoreable}
            best[regime_name] = max(scored, key=scored.get)
        rec["regime_best_action"] = best
        out.append(rec)
    return out
