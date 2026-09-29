
import pytest

from scripts.evaluation.efficiency import frontier_point, write_frontier_csv
from scripts.evaluation.evaluate_agent import score_run


@pytest.fixture(autouse=True)
def reset_frontier():
    import scripts.evaluation.efficiency as _eff
    _eff._frontier_points.clear()


def test_score_run_counts_cost():
    run = {
        "finished": True,
        "stop_reason": "done",
        "steps": 3,
        "final_answer": "fixed",
        "cost": {"input_tokens": 100, "output_tokens": 50, "tool_calls": 2,
                 "router_forwards": 3, "wall_ms": 1200, "total_cost": 151.06},
        "actions": ["TOOL", "ANSWER"],
    }
    scored = score_run(run, reference_answer="fixed")
    assert scored["success"] is True
    assert scored["total_cost"] == 151.06
    assert scored["tool_calls"] == 2
    assert scored["router_forwards"] == 3


def test_score_run_failure():
    run = {"finished": True, "stop_reason": "budget", "steps": 2, "final_answer": "wrong",
           "cost": {"input_tokens": 10, "output_tokens": 5, "tool_calls": 0,
                    "router_forwards": 2, "wall_ms": 100, "total_cost": 15.2},
           "actions": ["ANSWER"]}
    scored = score_run(run, reference_answer="fixed")
    assert scored["success"] is False


def test_frontier_point_dominance():
    cheap_ok = frontier_point(success_rate=0.8, total_cost=100.0)
    costly_ok = frontier_point(success_rate=0.8, total_cost=200.0)
    assert cheap_ok["pareto"] is True
    assert costly_ok["pareto"] is False


def test_write_frontier_csv(tmp_path):
    rows = [
        {"system": "always_answer", "success_rate": 0.5, "mean_cost": 80.0, "pareto": True},
        {"system": "adaptive_laya", "success_rate": 0.8, "mean_cost": 90.0, "pareto": True},
    ]
    path = write_frontier_csv(rows, tmp_path / "frontier.csv")
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    header = lines[0].split(",")
    assert header == ["system", "success_rate", "mean_cost", "pareto"]

