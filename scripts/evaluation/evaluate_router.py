from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.calibration import expected_calibration_error
from model.router import ACTIONS, AdaptiveRouter


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def evaluate_predictions(bundle: dict[str, Any], preds: dict[str, torch.Tensor]) -> dict[str, float]:
    TOOL_IDX = ACTIONS.index("TOOL")
    ASK_IDX = ACTIONS.index("ASK")
    y = bundle["action"].numpy()
    p = preds["action"].numpy()
    tp_tool = int(((y == TOOL_IDX) & (p == TOOL_IDX)).sum())
    fn_tool = int(((y == TOOL_IDX) & (p != TOOL_IDX)).sum())
    tp_ask = int(((y == ASK_IDX) & (p == ASK_IDX)).sum())
    fp_ask = int(((y != ASK_IDX) & (p == ASK_IDX)).sum())
    tool_recall = tp_tool / (tp_tool + fn_tool) if tp_tool + fn_tool else 0.0
    ask_precision = tp_ask / (tp_ask + fp_ask) if tp_ask + fp_ask else 0.0
    conf = preds["confidence"].float()
    correct = torch.tensor(y == p, dtype=torch.float32)
    hn_mask = bundle.get("hard_negative")
    hn_acc = (
        float((p[hn_mask.numpy()] == y[hn_mask.numpy()]).mean())
        if hn_mask is not None and hn_mask.any() else None
    )
    utility = float((y == p).mean()) - 0.02 * float((p == 1).mean())
    return {
        "macro_f1": macro_f1(y, p, n_classes=3),
        "tool_recall": float(tool_recall),
        "ask_precision": float(ask_precision),
        "hard_negative_accuracy": hn_acc,
        "ece": expected_calibration_error(conf, correct),
        "utility": utility,
        "accuracy": float((y == p).mean()),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        stats.append(sample.mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_gates(metrics: dict[str, float], gates: dict[str, float]) -> dict[str, Any]:
    checks = {
        "macro_f1": metrics["macro_f1"] >= gates["macro_f1"],
        "hard_negative_accuracy": (metrics["hard_negative_accuracy"] is not None and metrics["hard_negative_accuracy"] >= gates["hard_negative_accuracy"])
            if metrics["hard_negative_accuracy"] is not None else True,
        "ece": metrics["ece"] <= gates["ece_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def predict_with_router(
    bundle: dict[str, Any], router: AdaptiveRouter, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    router.eval()
    with torch.no_grad():
        feats = bundle["features"].to(device)
        budget = torch.full((len(feats), 4), 0.5, device=device)
        out = router(feats, budget)
        actions = out.action_logits.argmax(-1).cpu()
        tools = out.tool_logits.argmax(-1).cpu()
        terminals = out.terminal_logits.argmax(-1).cpu()
        confs = out.confidence.cpu()
    tools = torch.where(actions == 1, tools, torch.full_like(tools, -1))
    return {"action": actions, "tool": tools, "terminal": terminals, "confidence": confs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--router", type=str, required=True)
    ap.add_argument("--router-config", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/router_eval.json")
    ap.add_argument("--gates", type=str, default="configs/phase0_qwen.yaml")
    args = ap.parse_args()

    import yaml

    bundle = torch.load(args.features, weights_only=False)
    cfg = json.loads(Path(args.router_config).read_text(encoding="utf-8"))
    router = AdaptiveRouter(hidden_size=cfg["hidden_size"])
    router.load_state_dict(torch.load(args.router, weights_only=True))
    preds = predict_with_router(bundle, router)
    metrics = evaluate_predictions(bundle, preds)

    gates_cfg = yaml.safe_load(Path(args.gates).read_text(encoding="utf-8"))["eval"]
    gate_result = run_gates(metrics, gates_cfg["gates"])
    acc = (preds["action"] == bundle["action"]).float().numpy()
    lo, hi = bootstrap_ci(acc, n_boot=gates_cfg.get("bootstrap_samples", 1000))
    gate_result["accuracy_ci95"] = [lo, hi]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")
    print(json.dumps(gate_result, indent=2))


if __name__ == "__main__":
    main()
