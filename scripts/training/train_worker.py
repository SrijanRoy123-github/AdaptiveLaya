from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_generation.ingest_real import ingest
from scripts.kaggle_io import download, publish
from scripts.publish_checkpoint import build_manifest
from scripts.training.train_router import train_router


def worker_config(workers_yaml: str, worker_id: int) -> dict:
    cfg = yaml.safe_load(Path(workers_yaml).read_text(encoding="utf-8"))
    return next(w for w in cfg["workers"] if w["id"] == worker_id)


def resume_from_merged(out_dir: Path, merged_slug: str) -> None:
    if (out_dir / "router.pt").exists():
        return
    try:
        merged = download(merged_slug, dst=out_dir / "resume")
    except Exception as exc:  # noqa: BLE001 - merged dataset may not exist yet (remote 404 etc.)
        print(f"[resume] skipped {merged_slug}: {exc}")
        return
    import shutil

    for item in Path(merged).iterdir():
        target = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def run_probe(features_path: Path, out_dir: Path, worker: dict, round_idx: int,
              base_revision: str, hidden_size: int, epochs: int, lr: float) -> dict:
    if not features_path.exists():
        raise SystemExit(
            f"features file not found: {features_path} - run probe_decoder.py first (probe cell failed?)"
        )
    bundle = torch.load(features_path, weights_only=False)
    coeffs = {"lambda_action": 1.0, "lambda_tool": 1.0, "lambda_terminal": 1.0,
              "lambda_conf": 0.5, "lambda_budget": 0.0}
    metrics = train_router(
        bundle, hidden_size=hidden_size, epochs=epochs, lr=lr, coeffs=coeffs,
        out_dir=out_dir, seed=worker["seed"],
    )
    manifest = build_manifest(worker["id"], round_idx, worker["seed"], step=epochs,
                                base_revision=base_revision, extra={"mode": "probe", **metrics})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return metrics


def run_lora(shard_path: Path, out_dir: Path, worker: dict, round_idx: int,
             base_revision: str, cfg: dict, hours_limit: float) -> dict:
    from model.backbone import load_decoder
    from scripts.training.train_lora import train_lora_on_model

    started = time.monotonic()
    model, tok = load_decoder(cfg["backbone"]["model_id"], device="cuda")
    metrics = train_lora_on_model(
        model, tok, shard_path, out_dir,
        lora_cfg=cfg["lora"], router_cfg=cfg["router"], loss_cfg=cfg["loss"],
        train_cfg=cfg["training"], seed=worker["seed"], device="cuda",
        hours_limit=hours_limit,
    )
    manifest = build_manifest(
        worker["id"], round_idx, worker["seed"], step=int(metrics.get("steps", 0)),
        base_revision=base_revision,
        extra={"mode": "lora", "elapsed_s": time.monotonic() - started, **metrics},
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--mode", type=str, default="probe", choices=["probe", "lora"])
    ap.add_argument("--features", type=str, default=None, help="probe mode: features .pt")
    ap.add_argument("--shard", type=str, default=None, help="lora mode: shard jsonl")
    ap.add_argument("--out", type=str, default="results/worker_out")
    ap.add_argument("--config", type=str, default="configs/phase2_lora.yaml")
    ap.add_argument("--workers-yaml", type=str, default="configs/worker.yaml")
    ap.add_argument("--base-revision", type=str, default="pinned-initial")
    ap.add_argument("--hidden-size", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--skip-ingest", action="store_true",
                    help="skip dataset download (data already ingested by notebook)")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    workers_cfg = yaml.safe_load(Path(args.workers_yaml).read_text(encoding="utf-8"))
    worker = worker_config(args.workers_yaml, args.worker_id)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_slug = workers_cfg["datasets"]["merged"]

    resume_from_merged(out_dir, merged_slug)

    if not args.skip_ingest:
        for ds in workers_cfg.get("datasets", {}).get("real", []):
            ingest(ds)

    if args.mode == "probe":
        if not args.features:
            raise SystemExit("--features required in probe mode")
        metrics = run_probe(Path(args.features), out_dir, worker, args.round,
                            args.base_revision, args.hidden_size, args.epochs, args.lr)
    else:
        if not args.shard:
            raise SystemExit("--shard required in lora mode")
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        hours_limit = float(os.environ.get("ADAPTIVE_LAYA_SESSION_HOURS", "9.5"))
        metrics = run_lora(Path(args.shard), out_dir, worker, args.round,
                            args.base_revision, cfg, hours_limit)

    if args.publish:
        from scripts.verify_checkpoint import verify_checkpoint

        result = verify_checkpoint(out_dir, args.base_revision)
        if not result["ok"]:
            raise SystemExit(json.dumps(result, indent=2))
        publish(out_dir, worker["artifact_slug"], f"round {args.round} worker {args.worker_id}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()