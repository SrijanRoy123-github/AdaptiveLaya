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
