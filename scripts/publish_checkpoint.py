from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.kaggle_io import publish
from scripts.verify_checkpoint import verify_checkpoint


def build_manifest(
    worker_id: int,
    round_idx: int,
    seed: int,
    step: int,
    base_revision: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "experiment": "adaptive-laya-600m",
        "worker_id": worker_id,
        "round": round_idx,
        "seed": seed,
        "step": step,
        "base_revision": base_revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--slug", type=str, required=True)
    ap.add_argument("--message", type=str, default="adaptive-laya checkpoint")
    ap.add_argument("--base-revision", type=str, default=None)
    args = ap.parse_args()

    result = verify_checkpoint(args.src, args.base_revision)
    if not result["ok"]:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    publish(args.src, args.slug, args.message)
    print("published", args.src, "->", args.slug)


if __name__ == "__main__":
    main()
