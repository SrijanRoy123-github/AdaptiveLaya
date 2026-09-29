from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/worker.yaml")


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def shard_of(record_id: str, n_shards: int) -> int:
    """Stable hash bucket for record_id. NOT the production membership function —
    actual shard membership is assigned only by shard_records() (seeded stratified
    round-robin) and recorded in the output files / shards.manifest.json."""
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    return _stable_hash(record_id) % n_shards


def shard_records(records: list[dict[str, Any]], n: int, seed: int = 42) -> list[list[dict[str, Any]]]:
    if n < 1:
        raise ValueError(f"n_shards must be >= 1, got {n}")
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        strata[rec["label_action"]].append(rec)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    rng = random.Random(seed)
    for key in sorted(strata):
        group = strata[key]
        group.sort(key=lambda r: _stable_hash(r["id"]))
        rng.shuffle(group)
        for i, rec in enumerate(group):
            shards[i % n].append(rec)
    for s in shards:
        s.sort(key=lambda r: r["id"])
    return shards


def _holdout_counts(n_label: int, val_frac: float, test_frac: float) -> tuple[int, int]:
    # half-up rounding so frac * n_label >= 0.5 yields >= 1
    n_val = min(int(n_label * val_frac + 0.5), n_label)
    n_test = min(int(n_label * test_frac + 0.5), n_label - n_val)
    return n_val, n_test


def split_shared_holdouts(
    records: list[dict[str, Any]],
    val_frac: float,
    test_frac: float,
    seed: int = 0,
) -> tuple[list, list, list]:
    """Stratified shared holdouts: split WITHIN each label_action stratum so
    val/test preserve label balance. Returns (val, test, train)."""
    if val_frac < 0 or test_frac < 0:
        raise ValueError(f"fractions must be non-negative, got val_frac={val_frac}, test_frac={test_frac}")
    if val_frac + test_frac >= 1:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac + test_frac}")
    rng = random.Random(seed)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        strata[rec["label_action"]].append(rec)
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: _stable_hash(r["id"]))
        rng.shuffle(group)
        n_label = len(group)
        n_val, n_test = _holdout_counts(n_label, val_frac, test_frac)
        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])
    return val, test, train


def _holdout_config_defaults(config_path: Path = CONFIG_PATH) -> tuple[float, float]:
    try:
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return float(cfg["shared_validation_fraction"]), float(cfg["shared_test_fraction"])
    except (OSError, yaml.YAMLError, KeyError):
        return 0.1, 0.1


def _payload(records: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(r) for r in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    cfg_val, cfg_test = _holdout_config_defaults(CONFIG_PATH)
    ap.add_argument("--input", type=str, default="data/decisions/decisions_v1/decisions.jsonl")
    ap.add_argument("--out", type=str, default="data/processed/shards")
    ap.add_argument("--n", type=int, default=3, help="number of shards")
    ap.add_argument("--seed", type=int, default=42, help="seed for stratified shard assignment")
    ap.add_argument(
        "--val-frac", type=float, default=cfg_val,
        help="shared validation fraction (default: configs/worker.yaml shared_validation_fraction, else 0.1)",
    )
    ap.add_argument(
        "--test-frac", type=float, default=cfg_test,
        help="shared test fraction (default: configs/worker.yaml shared_test_fraction, else 0.1)",
    )
    ap.add_argument("--holdout-seed", type=int, default=0, help="seed for shared holdout split")
    args = ap.parse_args()

    if args.n < 1:
        ap.error(f"--n must be >= 1, got {args.n}")
    in_path = Path(args.input)
    if not in_path.is_file():
        ap.error(f"input file not found: {args.input}")
    raw = in_path.read_bytes()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except json.JSONDecodeError as e:
        ap.error(f"invalid JSONL in {args.input}: {e}")
    try:
        val, test, train = split_shared_holdouts(records, args.val_frac, args.test_frac, seed=args.holdout_seed)
    except ValueError as e:
        ap.error(str(e))

    shards = shard_records(train, args.n, seed=args.seed)
    payloads: dict[str, bytes] = {}
    for name, split in (("val.jsonl", val), ("test.jsonl", test)):
        payloads[name] = _payload(split)
    for i, shard in enumerate(shards):
        payloads[f"shard-{i}.jsonl"] = _payload(shard)
    files = {name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in payloads.items():
        (out / name).write_bytes(blob)
    manifest = {
        "input_sha256": input_sha256,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_shards": args.n,
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "shards": [len(s) for s in shards],
        },
        "files": files,
    }
    # manifest written LAST: readers treat a missing/mismatched manifest as stale
    (out / "shards.manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"train={len(train)} val={len(val)} test={len(test)} shards={[len(s) for s in shards]}")


if __name__ == "__main__":
    main()
