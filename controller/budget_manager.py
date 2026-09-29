from __future__ import annotations

import math
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
        return max(int(self.max_latency_ms - self.used_latency_ms), 0)

    def vector(self) -> list[float]:
        return [
            min(max(self.remaining_tokens / max(self.max_tokens, 1), 0.0), 1.0),
            min(max(self.remaining_tools / max(self.max_tools, 1), 0.0), 1.0),
            min(max(self.remaining_latency_ms / max(self.max_latency_ms, 1), 0.0), 1.0),
            min(max(self.step / max(self.max_steps, 1), 0.0), 1.0),
        ]

    def can_spend_tokens(self, n: int) -> bool:
        return self.used_tokens + int(n) <= self.max_tokens

    def can_call_tool(self) -> bool:
        return self.used_tools < self.max_tools

    def record_tokens(self, n: int) -> None:
        n = int(n)
        if n < 0:
            raise ValueError(f"tokens must be non-negative, got {n}")
        self.used_tokens += n

    def record_tool(self) -> None:
        self.used_tools += 1

    def record_latency(self, ms: float) -> None:
        ms = float(ms)
        if not math.isfinite(ms) or ms < 0:
            raise ValueError(f"latency must be non-negative finite, got {ms!r}")
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
