from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.losses import router_loss
from model.router import AdaptiveRouter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def budget_batch(n: int) -> torch.Tensor:
    return torch.full((n, 4), 0.5)


def train_router(
    bundle: dict[str, Any],
    hidden_size: int,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    coeffs: dict[str, float] | None = None,
    out_dir: str | Path = "results/router_run",
    seed: int = 42,
    device: str | None = None,
    val_frac: float = 0.1,
) -> dict[str, float]:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    coeffs = coeffs or {"lambda_action": 1.0, "lambda_tool": 1.0,
                        "lambda_terminal": 1.0, "lambda_conf": 0.5, "lambda_budget": 0.0}

    feats = bundle["features"]
    n = feats.shape[0]
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_size = max(1, int(n * val_frac))
    val_set = set(idx[:val_size])
    train_idx = [i for i in idx[val_size:]]
    if not train_idx:
        raise ValueError("not enough samples for train/val split")
    train_idx_t = torch.tensor(sorted(train_idx))

    ds = TensorDataset(
        feats[train_idx_t],
        bundle["action"][train_idx_t],
        bundle["tool"][train_idx_t],
        bundle["terminal"][train_idx_t],
        bundle["confidence"][train_idx_t],
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    router = AdaptiveRouter(hidden_size=hidden_size).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)

    first_loss = None
    final_loss = None
    val_losses = []
    val_idx_t = torch.tensor(sorted(val_set))
    val_feats = feats[val_idx_t].to(device)
    val_action = bundle["action"][val_idx_t].to(device)
    val_tool = bundle["tool"][val_idx_t].to(device)
    val_terminal = bundle["terminal"][val_idx_t].to(device)
    val_conf = bundle["confidence"][val_idx_t].to(device)
    for _ in range(epochs):
        for b_feats, b_action, b_tool, b_term, b_conf in loader:
            b_feats = b_feats.to(device)
            out = router(b_feats, budget_batch(len(b_feats)).to(device))
            labels = {
                "action": b_action.to(device),
                "tool": b_tool.to(device),
                "terminal": b_term.to(device),
                "confidence": b_conf.to(device),
            }
            parts = router_loss(out, labels, coeffs)
            loss = parts["loss"]
            if first_loss is None:
                first_loss = float(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
        router.eval()
        with torch.no_grad():
            v_out = router(val_feats, budget_batch(len(val_feats)).to(device))
            v_labels = {
                "action": val_action,
                "tool": val_tool,
                "terminal": val_terminal,
                "confidence": val_conf,
            }
            v_parts = router_loss(v_out, v_labels, coeffs)
            val_losses.append(float(v_parts["loss"].item()))
        router.train()
    val_loss = float(np.mean(val_losses)) if val_losses else 0.0

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), out_path / "router.pt")
    (out_path / "router_config.json").write_text(
        json.dumps({"hidden_size": hidden_size, "seed": seed, "epochs": epochs,
                    "batch_size": batch_size, "lr": lr}, indent=2)
    )
    metrics = {"first_loss": first_loss, "final_loss": final_loss,
               "val_loss": val_loss,
               "train_size": len(train_idx), "val_size": len(val_set)}
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/router_run")
    ap.add_argument("--hidden-size", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml

    bundle = torch.load(args.features, weights_only=False)
    hidden = args.hidden_size
    cfg_path = Path("configs/phase0_qwen.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        hidden = int(cfg.get("router", {}).get("hidden_size", hidden))
    coeffs = {"lambda_action": 1.0, "lambda_tool": 1.0, "lambda_terminal": 1.0,
              "lambda_conf": 0.5, "lambda_budget": 0.0}
    metrics = train_router(bundle, hidden, args.epochs, args.batch_size, args.lr,
                           coeffs, args.out, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
