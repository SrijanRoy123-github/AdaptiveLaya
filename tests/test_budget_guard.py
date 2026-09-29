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


def test_vector_matches_state_serializer():
    import yaml

    from controller.state_serializer import budget_vector, build_state

    regimes = yaml.safe_load((ROOT / "configs" / "regimes.yaml").read_text(encoding="utf-8"))["regimes"]
    regime = regimes["NORMAL"]
    mgr = BudgetManager(regime)
    mgr.record_tokens(100)
    mgr.record_tool()
    mgr.record_latency(5000)
    mgr.record_step()
    state = build_state(
        {"step_index": mgr.step},
        remaining_tokens=mgr.remaining_tokens,
        remaining_tools=mgr.remaining_tools,
        remaining_latency_ms=mgr.remaining_latency_ms,
    )
    assert mgr.vector() == budget_vector(state, regime)


def test_record_tokens_rejects_negative():
    mgr = BudgetManager(load_regime(ROOT / "configs" / "regimes.yaml", "NORMAL"))
    try:
        mgr.record_tokens(-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_record_latency_rejects_nan():
    mgr = BudgetManager(load_regime(ROOT / "configs" / "regimes.yaml", "NORMAL"))
    try:
        mgr.record_latency(float("nan"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_exhausted_when_budget_spent():
    regime = {"max_tokens": 1, "max_tools": 1, "max_latency_ms": 1, "max_steps": 1}
    mgr = BudgetManager(regime)
    assert mgr.exhausted() is False
    mgr.record_tokens(1)
    mgr.record_step()
    assert mgr.exhausted() is True


def test_vector_clamps_after_overspend_guard():
    regime = {"max_tokens": 10, "max_tools": 1, "max_latency_ms": 100, "max_steps": 2}
    mgr = BudgetManager(regime)
    mgr.record_tokens(50)
    v = mgr.vector()
    assert v[0] == 0.0
    assert all(0.0 <= x <= 1.0 for x in v)


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
