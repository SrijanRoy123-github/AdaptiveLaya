from __future__ import annotations

import time
from typing import Any, Protocol

from controller.budget_manager import BudgetManager
from controller.context_manager import format_observation
from controller.cost_ledger import CostLedger
from controller.loop_guard import LoopGuard
from controller.state_serializer import build_state, serialize_state
from tools.registry import execute


class RouterLike(Protocol):
    def decide(self, state: dict[str, Any], budget_vec: list[float]) -> dict[str, Any]: ...


class GeneratorLike(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, int]: ...


class MockGenerator:
    def __init__(self, text: str = "answer"):
        self.text = text

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, int, int]:
        words = self.text.split()
        return self.text, len(prompt.split()), len(words)


class ScriptedRouter:
    def __init__(self, decisions: list[dict[str, Any]]):
        self.decisions = decisions
        self.i = 0

    def decide(self, state: dict[str, Any], budget_vec: list[float]) -> dict[str, Any]:
        d = self.decisions[min(self.i, len(self.decisions) - 1)]
        self.i += 1
        return dict(d)


class AgentLoop:
    def __init__(
        self,
        router: RouterLike,
        generator: GeneratorLike,
        regime: dict[str, Any],
        loop_limits: dict[str, Any],
        repo_path: str,
        cost_coeffs: dict[str, float] | None = None,
        logger=None,
    ):
        self.router = router
        self.generator = generator
        self.regime = regime
        self.guard = LoopGuard(loop_limits)
        self.repo_path = repo_path
        self.ledger = CostLedger(cost_coeffs or {
            "alpha_tool_calls": 5e-2, "beta_router_forwards": 1e-3, "gamma_wall_ms": 1e-5,
        })
        self.log = logger or (lambda record: None)

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        budget = BudgetManager(self.regime)
        history_actions: list[str] = []
        previous_obs = ""
        test_results = ""
        final_answer = ""
        stop_reason = "max_steps"
        start = time.monotonic()
        step = 0
        generated_tokens = 0
        tool_calls = 0
        obs_tokens = 0

        while True:
            wall_ms = (time.monotonic() - start) * 1000
            if not self.guard.check(step, generated_tokens, tool_calls, obs_tokens, wall_ms):
                stop_reason = self.guard.reason() or "loop_guard"
                break

            state = build_state(
                {**task, "previous_actions": history_actions,
                 "previous_tool_result": previous_obs, "test_results": test_results,
                 "step_index": step, "previous_confidence": 0.5},
                remaining_tokens=budget.remaining_tokens,
                remaining_tools=budget.remaining_tools,
                remaining_latency_ms=budget.remaining_latency_ms,
            )
            state_text = serialize_state(state, self.regime)
            t0 = time.monotonic()
            decision = self.router.decide(state, budget.vector())
            router_ms = (time.monotonic() - t0) * 1000
            self.ledger.add(router_forwards=1, wall_ms=router_ms)

            action = decision["action"]
            history_actions.append(action)
            self.log({"step": step, "decision": decision, "wall_ms": wall_ms})

            if action == "ANSWER" and decision.get("terminal") == "DONE":
                prompt = f"{state_text}\nANSWER:"
                answer, in_tok, out_tok = self.generator.generate(prompt, budget.remaining_tokens)
                final_answer = answer
                self.ledger.add(input_tokens=in_tok, output_tokens=out_tok)
                budget.record_tokens(in_tok + out_tok)
                generated_tokens += out_tok
                stop_reason = "done"
                step += 1
                budget.record_step()
                break

            if action == "TOOL":
                if not budget.can_call_tool():
                    stop_reason = "budget"
                    budget.record_step()
                    break
                tool = decision.get("tool") or "READ_CONTEXT"
                t1 = time.monotonic()
                obs = execute(tool, {"path": ".", "pattern": "def ", "target": "."}, self.repo_path)
                tool_ms = (time.monotonic() - t1) * 1000
                self.ledger.add(tool_calls=1, wall_ms=tool_ms)
                budget.record_tool()
                budget.record_latency(tool_ms)
                tool_calls += 1
                raw = obs.get("output", "")
                formatted = format_observation(raw, max_tokens=max(self.regime.get("max_tokens", 750) // 2, 50))
                obs_tokens += len(formatted.split())
                previous_obs = formatted
                if tool == "RUN_TESTS":
                    test_results = raw[-1000:]

            if action == "ASK":
                self.ledger.add(input_tokens=0, output_tokens=0)
                final_answer = "CLARIFY: " + task.get("user_request", "")
                stop_reason = "asked"
                step += 1
                budget.record_step()
                break

            if budget.exhausted():
                stop_reason = "budget"
                step += 1
                budget.record_step()
                break

            budget.record_step()
            step += 1

        return {
            "finished": True,
            "steps": step,
            "final_answer": final_answer,
            "stop_reason": stop_reason,
            "cost": self.ledger.as_dict(),
            "actions": history_actions,
        }

