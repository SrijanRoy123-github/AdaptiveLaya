from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def validate_pair(ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> tuple[bool, str]:
    if set(ref) != set(other):
        return False, f"key mismatch: {sorted(set(ref) ^ set(other))}"
    for k, ref_v in ref.items():
        if ref_v.shape != other[k].shape:
            return False, f"shape mismatch at {k}: {tuple(ref_v.shape)} vs {tuple(other[k].shape)}"
        if torch.is_floating_point(ref_v) and (torch.isnan(ref_v).any() or torch.isinf(ref_v).any()):
            return False, f"non-finite values in {k}"
        if torch.is_floating_point(other[k]) and (torch.isnan(other[k]).any() or torch.isinf(other[k]).any()):
            return False, f"non-finite values in {k}"
    return True, "ok"


def average_router_states(
    states: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("no states to average")
    if weights is None:
        weights = [1.0] * len(states)
    total = float(sum(weights))
    acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in states[0].items()}
    for state, w in zip(states, weights):
        for k, v in state.items():
            acc[k] += v.float() * (w / total)
    return {k: v.to(states[0][k].dtype) for k, v in acc.items()}


def merge_full_states(
    base: dict[str, torch.Tensor],
    worker_states: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    if weights is None:
        weights = [1.0] * len(worker_states)
    total = float(sum(weights))
    merged = {}
    for k, base_v in base.items():
        delta = torch.zeros_like(base_v, dtype=torch.float32)
        for state, w in zip(worker_states, weights):
            delta += (state[k].float() - base_v.float()) * (w / total)
        merged[k] = (base_v.float() + delta).to(base_v.dtype)
    return merged


def select_best_worker(workers: list[dict[str, Any]]) -> dict[str, Any]:
    return max(workers, key=lambda w: w.get("validation_score", float("-inf")))


def validation_weighted(workers: list[dict[str, Any]]) -> list[float]:
    scores = [max(w.get("validation_score", 0.0), 1e-6) for w in workers]
    total = sum(scores)
    return [s / total for s in scores]


def aggregate_checkpoints(
    worker_dirs: list[Path],
    method: str,
    out_dir: Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    states = []
    metas = []
    for d in worker_dirs:
        router = torch.load(d / "router.pt", map_location="cpu", weights_only=True)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        ref = states[0] if states else router
        ok, msg = validate_pair(ref, router)
        if not ok:
            raise ValueError(f"{d}: {msg}")
        states.append(router)
        if "validation_score" in manifest:
            score = float(manifest["validation_score"])
        elif "val_loss" in manifest:
            score = 1.0 / (1.0 + float(manifest["val_loss"]))
        else:
            score = 0.0
        metas.append({"dir": str(d), "manifest": manifest, "validation_score": score})

    if method == "best_worker":
        best = select_best_worker(metas)
        merged = torch.load(Path(best["dir"]) / "router.pt", map_location="cpu", weights_only=True)
        weights = [1.0 if m["dir"] == best["dir"] else 0.0 for m in metas]
    elif method == "validation_weighted":
        weights = validation_weighted(metas)
        merged = average_router_states(states, weights)
    else:
        weights = [1.0] * len(states)
        merged = average_router_states(states, weights)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_dir / "router.pt")
    config = json.loads((Path(worker_dirs[0]) / "router_config.json").read_text(encoding="utf-8"))
    (out_dir / "router_config.json").write_text(json.dumps(config, indent=2))
    manifest = {
        "experiment": "adaptive-laya-600m",
        "method": method,
        "weights": weights,
        "workers": [str(d) for d in worker_dirs],
        "round": metas[0]["manifest"].get("round"),
        "base_revision": metas[0]["manifest"].get("base_revision"),
        "seed": metas[0]["manifest"].get("seed"),
        "step": metas[0]["manifest"].get("step"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=str, nargs="+", required=True)
    ap.add_argument("--method", type=str, default="equal_weight",
                    choices=["best_worker", "equal_weight", "validation_weighted"])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    manifest = aggregate_checkpoints([Path(d) for d in args.workers], args.method, Path(args.out))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
