from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_cost_coeffs(path: str | Path) -> dict[str, float]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in cfg["cost_ledger"].items()}


class CostLedger:
    def __init__(self, coeffs: dict[str, float]):
        self.coeffs = {
            "alpha_tool_calls": float(coeffs["alpha_tool_calls"]),
            "beta_router_forwards": float(coeffs["beta_router_forwards"]),
            "gamma_wall_ms": float(coeffs["gamma_wall_ms"]),
        }
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
        wall = float(wall_ms)
        if not math.isfinite(wall) or wall < 0:
            raise ValueError(f"wall_ms must be a non-negative finite number, got {wall_ms!r}")
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.tool_calls += int(tool_calls)
        self.router_forwards += int(router_forwards)
        self.wall_ms += wall

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
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = 0
        self.router_forwards = 0
        self.wall_ms = 0.0
