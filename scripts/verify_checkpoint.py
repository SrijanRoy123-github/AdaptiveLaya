from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

REQUIRED_ROUTER_FILES = {"router.pt", "router_config.json", "manifest.json"}


def verify_checkpoint(ckpt_dir: str | Path, expected_base_revision: str | None = None) -> dict[str, Any]:
    root = Path(ckpt_dir)
    problems = []
    present = {p.name for p in root.iterdir()} if root.exists() else set()
    missing = REQUIRED_ROUTER_FILES - present
    if missing:
        problems.append(f"missing files: {sorted(missing)}")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_base_revision and manifest.get("base_revision") != expected_base_revision:
            problems.append("base_revision mismatch")
        for key in ("round", "seed", "step"):
            if key not in manifest:
                problems.append(f"manifest missing {key}")
        if "workers" in manifest:
            if not manifest["workers"]:
                problems.append("merged manifest has empty workers list")
        elif "worker_id" not in manifest:
            problems.append("manifest missing worker_id")
    router_path = root / "router.pt"
    if router_path.exists():
        try:
            state = torch.load(router_path, map_location="cpu", weights_only=True)
            for k, v in state.items():
                if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                    problems.append(f"non-finite values in {k}")
                    break
        except (OSError, RuntimeError) as exc:
            problems.append(f"router load failed: {exc}")
    adapter_path = root / "adapter_config.json"
    if adapter_path.exists():
        cfg = json.loads(adapter_path.read_text(encoding="utf-8"))
        if cfg.get("peft_type") != "LORA":
            problems.append("adapter is not LoRA")
    return {"ok": not problems, "problems": problems, "files": sorted(present)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--base-revision", type=str, default=None)
    args = ap.parse_args()
    result = verify_checkpoint(args.ckpt_dir, args.base_revision)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
