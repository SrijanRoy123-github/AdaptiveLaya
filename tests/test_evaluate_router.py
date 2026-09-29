import torch

from model.router import TOOLS
from scripts.evaluation.baselines import always_answer, always_tool, keyword_router
from scripts.evaluation.evaluate_router import (
    bootstrap_ci,
    evaluate_predictions,
    run_gates,
)


def make_eval_set(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    bundle = {
        "features": torch.randn(n, 32, generator=g),
        "action": torch.randint(0, 3, (n,), generator=g),
        "tool": torch.randint(0, len(TOOLS), (n,), generator=g),
        "terminal": torch.randint(0, 3, (n,), generator=g),
        "confidence": torch.rand(n, generator=g),
        "hard_negative": torch.zeros(n, dtype=torch.bool),
        "ids": [f"e{i}" for i in range(n)],
    }
    bundle["hard_negative"][::5] = True
    return bundle


def test_evaluate_predictions_perfect():
    bundle = make_eval_set()
    preds = {
        "action": bundle["action"].clone(),
        "tool": bundle["tool"].clone(),
        "terminal": bundle["terminal"].clone(),
        "confidence": bundle["confidence"].clone(),
    }
    metrics = evaluate_predictions(bundle, preds)
    assert metrics["macro_f1"] > 0.9
    assert 0.0 <= metrics["ece"] <= 1.0
    assert "tool_recall" in metrics and "ask_precision" in metrics


def test_baselines():
    bundle = make_eval_set()
    assert always_answer(bundle)["action"].eq(0).all()
    assert always_tool(bundle)["action"].eq(1).all()
    kw = keyword_router(["run the tests please", "what does this do"])
    assert kw["action"][0] == 1
    assert kw["action"][1] == 0


def test_bootstrap_ci():
    values = torch.rand(100)
    lo, hi = bootstrap_ci(values.numpy(), n_boot=100, seed=0)
    assert lo <= hi


def test_gates_fail_and_pass():
    gates = {"macro_f1": 0.7, "hard_negative_accuracy": 0.75, "ece_max": 0.15}
    failing = {"macro_f1": 0.4, "hard_negative_accuracy": 0.5, "ece": 0.3}
    assert run_gates(failing, gates)["passed"] is False
    passing = {"macro_f1": 0.8, "hard_negative_accuracy": 0.8, "ece": 0.1}
    assert run_gates(passing, gates)["passed"] is True