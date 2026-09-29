from controller.agent_loop import AgentLoop, MockGenerator, ScriptedRouter
from controller.context_manager import format_observation


def test_format_observation_truncates():
    raw = "error line\n" * 5000
    out = format_observation(raw, max_tokens=100)
    assert len(out.split()) <= 100
    assert out.startswith("<|obs|>")
    assert out.endswith("<|obs_end|>")


def test_format_observation_preserves_errors():
    raw = "AssertionError: expected 3 got 4\n" + ("noise\n" * 1000)
    out = format_observation(raw, max_tokens=200)
    assert "AssertionError" in out


def test_agent_loop_terminates_with_done():
    router = ScriptedRouter([
        {"action": "TOOL", "tool": "READ_CONTEXT", "terminal": "CONTINUE", "confidence": 0.9},
        {"action": "ANSWER", "tool": None, "terminal": "DONE", "confidence": 0.9},
    ])
    loop = AgentLoop(
        router=router,
        generator=MockGenerator("fixed the bug"),
        regime={"max_tokens": 750, "max_tools": 2, "max_latency_ms": 60000, "max_steps": 8},
        loop_limits={"max_steps": 8, "max_tool_calls": 3, "max_generated_tokens": 1000,
                     "max_observation_tokens": 1000, "max_wall_time_ms": 60000},
        repo_path=".",
    )
    result = loop.run({"user_request": "fix bug", "repository_summary": "tiny repo"})
    assert result["finished"] is True
    assert result["steps"] == 2
    assert result["final_answer"] == "fixed the bug"
    assert result["cost"]["tool_calls"] >= 1


def test_agent_loop_budget_exhaustion():
    class AlwaysToolRouter:
        def decide(self, state, budget_vec):
            return {"action": "TOOL", "tool": "READ_CONTEXT", "terminal": "CONTINUE", "confidence": 0.9}

    loop = AgentLoop(
        router=AlwaysToolRouter(),
        generator=MockGenerator("x"),
        regime={"max_tokens": 50, "max_tools": 1, "max_latency_ms": 1000, "max_steps": 2},
        loop_limits={"max_steps": 8, "max_tool_calls": 3, "max_generated_tokens": 1000,
                     "max_observation_tokens": 1000, "max_wall_time_ms": 60000},
        repo_path=".",
    )
    result = loop.run({"user_request": "hi", "repository_summary": "r"})
    assert result["finished"] is True
    assert result["stop_reason"] in {"budget", "loop_guard", "max_steps"}

