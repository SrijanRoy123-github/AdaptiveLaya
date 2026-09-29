from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def benchmark(features: torch.Tensor, hidden_size: int, device: str, batch_size: int = 64) -> dict:
    from torch.utils.data import DataLoader, TensorDataset

    from model.router import AdaptiveRouter
    from scripts.training.train_router import budget_batch

    router = AdaptiveRouter(hidden_size=hidden_size).to(device)
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=True)
    n_tokens = 0
    n_samples = 0
    steps = 0
    started = time.monotonic()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            router(batch, budget_batch(len(batch)).to(device))
            n_samples += len(batch)
            n_tokens += batch.shape[0] * batch.shape[1]
            steps += 1
            if time.monotonic() - started > 10:
                break
    elapsed = time.monotonic() - started
    return {
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "samples_per_sec": n_samples / max(elapsed, 1e-6),
        "steps_per_sec": steps / max(elapsed, 1e-6),
        "router_params_m": sum(p.numel() for p in router.parameters()) / 1e6,
        "vram_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else 0.0,
        "wall_s": elapsed,
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--hidden-size", type=int, default=896)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="results/throughput.json")
    args = ap.parse_args()

    bundle = torch.load(args.features, weights_only=False)
    metrics = benchmark(bundle["features"], args.hidden_size, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()