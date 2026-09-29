from pathlib import Path

from controller.cost_ledger import CostLedger, load_cost_coeffs

ROOT = Path(__file__).resolve().parents[1]


def test_total_formula():
    coeffs = {"alpha_tool_calls": 0.05, "beta_router_forwards": 0.001, "gamma_wall_ms": 1e-5}
    led = CostLedger(coeffs)
    led.add(input_tokens=100, output_tokens=50, tool_calls=2, router_forwards=3, wall_ms=1000)
    expected = 150 + 0.05 * 2 + 0.001 * 3 + 1e-5 * 1000
    assert abs(led.total() - expected) < 1e-9


def test_accumulates():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    led.add(input_tokens=10)
    led.add(output_tokens=5)
    assert led.total() == 15
    d = led.as_dict()
    assert d["input_tokens"] == 10 and d["output_tokens"] == 5


def test_load_from_config():
    coeffs = load_cost_coeffs(ROOT / "configs" / "utility_config.yaml")
    assert "alpha_tool_calls" in coeffs


def test_utility():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    led.add(output_tokens=1000)
    u = led.utility(success=True, cfg={"success_reward": 1.0, "alpha_tokens": 2e-4,
                                       "beta_tool_calls": 0.05, "gamma_latency_ms": 1e-5,
                                       "delta_router_forwards": 1e-3})
    assert abs(u - (1.0 - 1000 * 2e-4)) < 1e-9


def test_missing_coefficient_raises():
    try:
        CostLedger({"alpha_tool_calls": 1.0})
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_add_rejects_negative_wall_ms():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    try:
        led.add(wall_ms=-1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_add_rejects_nan_wall_ms():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    try:
        led.add(wall_ms=float("nan"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_reset_clears_all_counters():
    led = CostLedger({"alpha_tool_calls": 0.0, "beta_router_forwards": 0.0, "gamma_wall_ms": 0.0})
    led.add(input_tokens=10, output_tokens=5, tool_calls=2, router_forwards=1, wall_ms=500)
    led.reset()
    d = led.as_dict()
    assert d["input_tokens"] == 0 and d["output_tokens"] == 0
    assert d["tool_calls"] == 0 and d["router_forwards"] == 0
    assert d["wall_ms"] == 0 and d["total_cost"] == 0.0
