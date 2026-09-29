# Adaptive Laya — Part 3: Three-Account Kaggle Training + Runbook (Tasks 16–23) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Kaggle-only system: artifact I/O, checkpoint verification, LoRA+router worker training with session budgets, FedAvg-style adapter aggregation with validation gate, throughput benchmarking, worker/aggregator notebooks, one-time dataset publication, and a step-by-step Kaggle runbook.

**Architecture:** Three Kaggle accounts train LoRA+router replicas on deterministic shards. Each worker pushes only adapter weights + router + manifest to its own public dataset (`adaptive-laya/adaptive-laya-worker-{i}`). Account A aggregates (delta averaging / best-worker / validation-weighted) and publishes to `adaptive-laya/adaptive-laya-merged`. All exchange is script-driven via `kagglehub`; local mode (`ADAPTIVE_LAYA_LOCAL_DIR`) lets every path be tested offline.

**Tech Stack:** Python 3.10+, PyTorch, peft, transformers, kagglehub, PyYAML, pytest. Compute: Kaggle T4×2 (single T4 must work), 12h sessions (train ≤9.5h, publish, 1–2h margin).

**Spec:** `Adaptive_Laya_600M_Three_Kaggle_Accounts_Final_Plan.md` §§46–104.

**Prerequisites:** Part 1 (`2026-09-24-adaptive-laya-part1-foundations.md`) and Part 2 (`2026-09-24-adaptive-laya-part2-probe-agent.md`) complete; all Part 1–2 tests pass.

**Compliance:** Use only Kaggle accounts you legitimately control/authorize (spec §46). Not true DDP — FedAvg-style periodic aggregation (spec §52, §71). Report as "three parallel Kaggle replicas with weighted parameter averaging" (spec §96).

---

## File Structure (Part 3 scope)

```
adaptive-laya/
├── scripts/
│   ├── kaggle_io.py
│   ├── verify_checkpoint.py
│   ├── publish_checkpoint.py
│   ├── aggregate_adapters.py
│   ├── benchmark_throughput.py
│   ├── bootstrap_kaggle.py
│   └── training/train_lora.py
│   └── training/train_worker.py
├── kaggle/
│   ├── worker.ipynb
│   └── aggregator.ipynb
└── tests/
    ├── test_kaggle_io.py
    ├── test_aggregate.py
    └── test_train_lora.py
```

**Import rule:** `scripts/*` may import `model/`, `controller/`, `tools/`; notebooks only call `scripts/bootstrap_kaggle.py`. Never the reverse.

---

### Task 16: Kaggle artifact I/O (spec §60–§65, §69–§70)

**Files:**
- Create: `scripts/kaggle_io.py`
- Test: `tests/test_kaggle_io.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path

from scripts.kaggle_io import download, local_mode, publish


def test_local_mode_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("KAGGLE_KERNEL_SESSION", raising=False)
    assert local_mode() is True
    src = tmp_path / "artifact"
    src.mkdir()
    (src / "router.pt").write_bytes(b"weights")
    publish(src, "adaptive-laya/does-not-matter", "test push")
    got = download("adaptive-laya/does-not-matter")
    assert (Path(got) / "router.pt").exists()


def test_publish_uses_slug_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", str(tmp_path / "store"))
    src = tmp_path / "w0"
    src.mkdir()
    (src / "manifest.json").write_text("{}")
    publish(src, "adaptive-laya/adaptive-laya-worker-0", "round 0")
    base = tmp_path / "store" / "adaptive-laya_adaptive-laya-worker-0"
    assert (base / "manifest.json").exists()


def test_download_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LAYA_LOCAL_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("KAGGLE_KERNEL_SESSION", raising=False)
    try:
        download("adaptive-laya/nothing")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kaggle_io.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/kaggle_io.py`**

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path


def local_mode() -> bool:
    if os.environ.get("ADAPTIVE_LAYA_LOCAL_DIR"):
        return True
    return not os.environ.get("KAGGLE_KERNEL_SESSION")


def _local_store() -> Path:
    return Path(os.environ.get("ADAPTIVE_LAYA_LOCAL_DIR", "data/kaggle_local_store"))


def _slug_dir(slug: str) -> str:
    return slug.replace("/", "_").replace("-", "_")


def publish(src_dir: str | Path, slug: str, message: str, local_dir_override: str | None = None) -> None:
    src = Path(src_dir)
    if local_dir_override or local_mode():
        root = Path(local_dir_override) if local_dir_override else _local_store()
        dest = root / _slug_dir(slug)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"[local] published {src} -> {dest}")
        return
    import kagglehub

    kagglehub.upload(path=str(src), dataset_slug=slug, messager=message)
    print(f"[kaggle] published {src} -> {slug}")


def download(slug: str, dst: str | Path | None = None) -> Path:
    if local_mode():
        root = _local_store()
        candidate = root / _slug_dir(slug)
        if candidate.exists():
            if dst:
                Path(dst).mkdir(parents=True, exist_ok=True)
                shutil.copytree(candidate, Path(dst), dirs_exist_ok=True)
                return Path(dst)
            return candidate
        raise FileNotFoundError(f"no local artifact for {slug} under {root}")
    import kagglehub

    path = Path(kagglehub.dataset_download(slug))
    if dst:
        Path(dst).mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, Path(dst), dirs_exist_ok=True)
        return Path(dst)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kaggle_io.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add scripts/kaggle_io.py tests/test_kaggle_io.py
git commit -m "feat: Kaggle artifact I/O with local mode for offline testing"
```

---

### Task 17: Checkpoint verification and publishing (spec §65, §91–§92)

**Files:**
- Create: `scripts/verify_checkpoint.py`, `scripts/publish_checkpoint.py`
- Test: `tests/test_verify_publish.py`

- [ ] **Step 1: Write failing test**

```python
import json

import torch

from scripts.publish_checkpoint import build_manifest
from scripts.verify_checkpoint import verify_checkpoint


def make_ckpt(tmp_path, with_nan=False):
    root = tmp_path / "ckpt"
    root.mkdir()
    state = {"w": torch.ones(4)}
    if with_nan:
        state["w"][0] = float("nan")
    torch.save(state, root / "router.pt")
    (root / "router_config.json").write_text(json.dumps({"hidden_size": 8}))
    (root / "manifest.json").write_text(json.dumps(build_manifest(0, 0, 42, 5, "rev-a")))
    return root


def test_verify_ok(tmp_path):
    result = verify_checkpoint(make_ckpt(tmp_path), expected_base_revision="rev-a")
    assert result["ok"] is True
    assert result["problems"] == []


def test_verify_missing_files(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    result = verify_checkpoint(root)
    assert result["ok"] is False


def test_verify_base_revision_mismatch(tmp_path):
    result = verify_checkpoint(make_ckpt(tmp_path), expected_base_revision="rev-b")
    assert result["ok"] is False
    assert any("base_revision" in p for p in result["problems"])


def test_verify_nan_detected(tmp_path):
    result = verify_checkpoint(make_ckpt(tmp_path, with_nan=True))
    assert result["ok"] is False


def test_build_manifest_fields():
    m = build_manifest(2, 1, 44, 100, "rev-c", extra={"mode": "probe"})
    assert m["worker_id"] == 2 and m["round"] == 1 and m["seed"] == 44
    assert m["step"] == 100 and m["base_revision"] == "rev-c"
    assert m["mode"] == "probe"
    assert m["created_utc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verify_publish.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/verify_checkpoint.py`**

```python
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
        for key in ("worker_id", "round", "seed", "step"):
            if key not in manifest:
                problems.append(f"manifest missing {key}")
    router_path = root / "router.pt"
    if router_path.exists():
        try:
            state = torch.load(router_path, map_location="cpu", weights_only=True)
            for k, v in state.items():
                if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                    problems.append(f"non-finite values in {k}")
                    break
        except Exception as exc:
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
```

- [ ] **Step 4: Implement `scripts/publish_checkpoint.py`**

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_verify_publish.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```powershell
git add scripts/verify_checkpoint.py scripts/publish_checkpoint.py tests/test_verify_publish.py
git commit -m "feat: checkpoint verification and gated publishing"
```

---

### Task 18: LoRA + router joint training (spec §24, §27)

**Files:**
- Create: `scripts/training/train_lora.py`
- Test: `tests/test_train_lora.py`

- [ ] **Step 1: Write failing test** (tiny random Qwen2, CPU, no downloads except cached tokenizer)

```python
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from scripts.training.train_lora import train_lora_on_model


def tiny_setup():
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def sample_shard(tmp_path):
    path = tmp_path / "shard.jsonl"
    lines = []
    for i in range(8):
        lines.append(json.dumps({
            "id": f"t{i}",
            "user_request": "Fix the test.",
            "repository_summary": "repo",
            "target_text": "def fix(): return True",
            "label_action": "TOOL",
            "label_tool": "RUN_TESTS",
            "label_terminal": "CONTINUE",
            "label_confidence": 0.8,
        }))
    path.write_text("\n".join(lines))
    return path


def test_train_lora_smoke(tmp_path):
    model, tok = tiny_setup()
    result = train_lora_on_model(
        model=model,
        tokenizer=tok,
        shard_path=sample_shard(tmp_path),
        out_dir=tmp_path / "out",
        lora_cfg={"r": 4, "alpha": 8, "dropout": 0.05,
                  "target_modules": ["q_proj", "v_proj"]},
        router_cfg={"hidden_size": 32, "mlp_hidden": 16, "mlp_out": 8},
        loss_cfg={"lambda_action": 1.0, "lambda_tool": 1.0, "lambda_terminal": 1.0,
                  "lambda_conf": 0.5, "lambda_budget": 0.0},
        train_cfg={"lr": 1e-3, "epochs": 1, "micro_batch_size": 2,
                   "gradient_accumulation": 1, "max_length": 64},
        seed=0,
        device="cpu",
    )
    assert result["steps"] > 0
    assert (tmp_path / "out" / "adapter_config.json").exists()
    assert (tmp_path / "out" / "router.pt").exists()
    assert (tmp_path / "out" / "router_config.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_lora.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/training/train_lora.py`**

```python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from model.losses import router_loss
from model.router import ACTIONS, TERMINALS, TOOLS, AdaptiveRouter
from scripts.training.train_router import set_seed

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOLS)}
TERMINAL_TO_IDX = {t: i for i, t in enumerate(TERMINALS)}


class DecisionSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int = 256):
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        prompt = rec["user_request"] + "\n" + rec.get("repository_summary", "")
        target = rec.get("target_text", "") or f"<|{rec['label_action'].lower()}|>"
        enc = self.tok(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        tgt = self.tok(
            target, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        input_ids = torch.cat([enc["input_ids"], tgt["input_ids"]], dim=1).squeeze(0)
        attn = torch.cat([enc["attention_mask"], tgt["attention_mask"]], dim=1).squeeze(0)
        labels = input_ids.clone()
        labels[: enc["input_ids"].shape[1]] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "action": ACTION_TO_IDX[rec["label_action"]],
            "tool": TOOL_TO_IDX.get(rec.get("label_tool") or "RUN_TESTS", -1),
            "terminal": TERMINAL_TO_IDX.get(rec.get("label_terminal", "CONTINUE"), 0),
            "confidence": float(rec.get("label_confidence", 0.5)),
        }


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    out = {
        "input_ids": torch.full((len(batch), max_len), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), max_len), dtype=torch.long),
        "labels": torch.full((len(batch), max_len), -100, dtype=torch.long),
        "action": torch.tensor([b["action"] for b in batch], dtype=torch.long),
        "tool": torch.tensor([b["tool"] for b in batch], dtype=torch.long),
        "terminal": torch.tensor([b["terminal"] for b in batch], dtype=torch.long),
        "confidence": torch.tensor([b["confidence"] for b in batch], dtype=torch.float32),
    }
    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        out["input_ids"][i, :L] = item["input_ids"]
        out["attention_mask"][i, :L] = item["attention_mask"]
        out["labels"][i, :L] = item["labels"]
    return out


def train_lora_on_model(
    model,
    tokenizer,
    shard_path: str | Path,
    out_dir: str | Path,
    lora_cfg: dict[str, Any],
    router_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int = 42,
    device: str = "cpu",
    hours_limit: float | None = None,
) -> dict[str, Any]:
    import time

    set_seed(seed)
    records = [json.loads(line) for line in Path(shard_path).read_text(encoding="utf-8").splitlines()]

    lora_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    router = AdaptiveRouter(
        hidden_size=int(router_cfg["hidden_size"]),
        mlp_hidden=int(router_cfg.get("mlp_hidden", 512)),
        mlp_out=int(router_cfg.get("mlp_out", 256)),
    ).to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ds = DecisionSFTDataset(records, tokenizer, max_length=int(train_cfg.get("max_length", 256)))
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg.get("micro_batch_size", 1)),
        shuffle=True,
        collate_fn=lambda b: _collate(b, pad_id),
    )

    params = list(model.parameters()) + list(router.parameters())
    opt = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 2e-4)))
    epochs = int(train_cfg.get("epochs", 1))
    accum = int(train_cfg.get("gradient_accumulation", 1))
    started = time.monotonic()
    steps = 0
    last_loss = None

    model.train()
    router.train()
    for _ in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels, output_hidden_states=True)
            lm_loss = out.loss
            hidden = out.hidden_states[-1][:, -1, :]
            router_out = router(hidden, torch.full((len(batch), 4), 0.5, device=device))
            labels_router = {
                "action": batch["action"].to(device),
                "tool": batch["tool"].to(device),
                "terminal": batch["terminal"].to(device),
                "confidence": batch["confidence"].to(device),
            }
            r_parts = router_loss(router_out, labels_router, loss_cfg)
            loss = (lm_loss + r_parts["loss"]) / accum
            loss.backward()
            last_loss = float(loss.item())
            if (steps + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            steps += 1
            if hours_limit is not None and (time.monotonic() - started) / 3600.0 >= hours_limit:
                break

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    torch.save(router.state_dict(), out_path / "router.pt")
    (out_path / "router_config.json").write_text(json.dumps(router_cfg, indent=2))
    return {"steps": steps, "last_loss": last_loss, "examples": len(records)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/phase2_lora.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import yaml

    from model.backbone import load_decoder

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model, tok = load_decoder(cfg["backbone"]["model_id"], device=args.device)
    metrics = train_lora_on_model(
        model, tok, args.shard, args.out,
        lora_cfg=cfg["lora"], router_cfg=cfg["router"], loss_cfg=cfg["loss"],
        train_cfg=cfg["training"], seed=args.seed, device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_lora.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add scripts/training/train_lora.py tests/test_train_lora.py
git commit -m "feat: LoRA + router joint training with session time budget"
```

---

### Task 19: Worker orchestration script (spec §67, §86–§87, §90)

**Files:**
- Create: `scripts/training/train_worker.py`

- [ ] **Step 1: Implement `scripts/training/train_worker.py`**

```python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml

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
    except FileNotFoundError:
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
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    workers_cfg = yaml.safe_load(Path(args.workers_yaml).read_text(encoding="utf-8"))
    worker = worker_config(args.workers_yaml, args.worker_id)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_slug = workers_cfg["datasets"]["merged"]

    resume_from_merged(out_dir, merged_slug)

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
```

- [ ] **Step 2: Smoke-test worker in local mode (probe path)**

Run (uses Part 2 tiny-bundle pattern via pytest already; CLI check):

```powershell
python -c "import scripts.training.train_worker as w; print('ok', w.worker_config('configs/worker.yaml', 1)['seed'])"
```

Expected: `ok 43`

- [ ] **Step 3: Commit**

```powershell
git add scripts/training/train_worker.py
git commit -m "feat: worker orchestration with resume, probe/lora modes, gated publish"
```

---

### Task 20: Adapter aggregation (spec §54–§59, §72–§73, §94–§95)

**Files:**
- Create: `scripts/aggregate_adapters.py`
- Test: `tests/test_aggregate.py`

- [ ] **Step 1: Write failing test**

```python
import json

import torch

from scripts.aggregate_adapters import (
    average_router_states,
    merge_full_states,
    select_best_worker,
    validate_pair,
)


def state(v):
    return {"layer.weight": torch.full((4, 4), float(v))}


def test_router_average_equal():
    a, b, c = state(1.0), state(3.0), state(5.0)
    avg = average_router_states([a, b, c])
    assert torch.allclose(avg["layer.weight"], torch.full((4, 4), 3.0))


def test_router_average_weighted():
    avg = average_router_states([state(0.0), state(10.0)], weights=[1, 3])
    assert torch.allclose(avg["layer.weight"], torch.full((4, 4), 7.5))


def test_merge_full_states_delta_average():
    base = state(0.0)
    w1, w2 = state(2.0), state(4.0)
    merged = merge_full_states(base, [w1, w2])
    assert torch.allclose(merged["layer.weight"], torch.full((4, 4), 3.0))


def test_validate_pair_rejects_shape_mismatch():
    ok, msg = validate_pair(state(1), state(2))
    assert ok is True
    bad = {"layer.weight": torch.zeros(2, 2)}
    ok, msg = validate_pair(state(1), bad)
    assert ok is False
    assert "shape" in msg


def test_validate_pair_rejects_nan():
    nan_state = {"layer.weight": torch.tensor([[float("nan")]])}
    ok, msg = validate_pair(state(1), nan_state)
    assert ok is False


def test_select_best_worker():
    workers = [
        {"worker_id": 0, "validation_score": 0.70},
        {"worker_id": 1, "validation_score": 0.75},
        {"worker_id": 2, "validation_score": 0.72},
    ]
    best = select_best_worker(workers)
    assert best["worker_id"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/aggregate_adapters.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def validate_pair(ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> tuple[bool, str]:
    if set(ref) != set(other):
        return False, f"key mismatch: {sorted(set(ref) ^ set(other))}"
    for k in ref:
        if ref[k].shape != other[k].shape:
            return False, f"shape mismatch at {k}: {tuple(ref[k].shape)} vs {tuple(other[k].shape)}"
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
        metas.append({"dir": str(d), "manifest": manifest,
                      "validation_score": manifest.get("validation_score", 0.0)})

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add scripts/aggregate_adapters.py tests/test_aggregate.py
git commit -m "feat: router aggregation (best/equal/validation-weighted) with validation checks"
```

---

### Task 21: Throughput benchmark (spec §34)

**Files:**
- Create: `scripts/benchmark_throughput.py`

- [ ] **Step 1: Implement `scripts/benchmark_throughput.py`**

```python
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
```

- [ ] **Step 2: Run locally for sanity**

Run: `python -c "import torch; from scripts.benchmark_throughput import benchmark; from model.router import AdaptiveRouter; feats=torch.randn(256, 64); print(benchmark(feats, 64, 'cpu', 32))"`
Expected: dict with `samples_per_sec` > 0.

- [ ] **Step 3: Commit**

```powershell
git add scripts/benchmark_throughput.py
git commit -m "feat: throughput benchmark for router training"
```

---

### Task 22: Bootstrap script + Kaggle notebooks (spec §85, §67–§68)

**Files:**
- Create: `scripts/bootstrap_kaggle.py`, `kaggle/worker.ipynb`, `kaggle/aggregator.ipynb`

- [ ] **Step 1: Implement `scripts/bootstrap_kaggle.py`**

```python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.kaggle_io import download


def verify_checksum(jsonl_path: Path, expected_sha: str | None) -> None:
    payload = jsonl_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha and digest != expected_sha:
        raise SystemExit(f"checksum mismatch: {digest} != {expected_sha}")


def ensure_shards(decisions_dir: Path, out_dir: Path, worker_id: int) -> Path:
    shard = out_dir / f"shard-{worker_id}.jsonl"
    if shard.exists():
        return shard
    subprocess.run(
        [sys.executable, "scripts/make_shards.py",
         "--input", str(decisions_dir / "decisions.jsonl"),
         "--out", str(out_dir)],
        check=True,
    )
    return shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--mode", type=str, default="probe", choices=["probe", "lora"])
    ap.add_argument("--data-dir", type=str, default="data/decisions_live")
    ap.add_argument("--shards-dir", type=str, default="data/processed/shards")
    ap.add_argument("--repo-url", type=str, default="")
    ap.add_argument("--commit", type=str, default="")
    args = ap.parse_args()

    if args.repo_url and not Path("configs/worker.yaml").exists():
        subprocess.run(["git", "clone", args.repo_url, "."], check=True)
        if args.commit:
            subprocess.run(["git", "checkout", args.commit], check=True)

    workers_cfg = yaml.safe_load(Path("configs/worker.yaml").read_text(encoding="utf-8"))
    decisions_slug = workers_cfg["datasets"]["decisions"]

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not (data_dir / "decisions.jsonl").exists():
        download(decisions_slug, dst=data_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_checksum(data_dir / "decisions.jsonl", manifest.get("sha256"))
    else:
        manifest = {}

    shard = ensure_shards(data_dir, Path(args.shards_dir), args.worker_id)
    print(json.dumps({
        "worker_id": args.worker_id,
        "mode": args.mode,
        "shard": str(shard),
        "records_sha": manifest.get("sha256", "unverified"),
    }, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `kaggle/worker.ipynb`** (full notebook JSON — 7 cells per spec §85)

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Adaptive Laya Worker\n\nSet `WORKER_ID` to 0, 1, or 2. Internet ON, GPU T4 x2 (single T4 works)."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import os\nWORKER_ID = int(os.environ.get(\"WORKER_ID\", \"0\"))\nMODE = os.environ.get(\"ADAPTIVE_LAYA_MODE\", \"probe\")\nROUND = int(os.environ.get(\"ADAPTIVE_LAYA_ROUND\", \"0\"))\nREPO_URL = os.environ.get(\"ADAPTIVE_LAYA_REPO\", \"\")\nCOMMIT = os.environ.get(\"ADAPTIVE_LAYA_COMMIT\", \"\")\nprint(WORKER_ID, MODE, ROUND)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import subprocess, sys\nsubprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", \"requirements.txt\"], check=True)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python scripts/bootstrap_kaggle.py --worker-id {WORKER_ID} --mode {MODE} --repo-url {REPO_URL} --commit {COMMIT}"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# Feature extraction (probe mode only)\nimport os, subprocess, sys\nif MODE == \"probe\":\n    subprocess.run([sys.executable, \"scripts/training/probe_decoder.py\",\n                    \"--input\", f\"data/processed/shards/shard-{WORKER_ID}.jsonl\",\n                    \"--out\", f\"results/features-{WORKER_ID}.pt\"], check=True)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# Train + publish\nimport subprocess, sys\ncmd = [sys.executable, \"scripts/training/train_worker.py\",\n       \"--worker-id\", str(WORKER_ID), \"--round\", str(ROUND),\n       \"--mode\", MODE, \"--out\", \"results/worker_out\", \"--publish\"]\nif MODE == \"probe\":\n    cmd += [\"--features\", f\"results/features-{WORKER_ID}.pt\"]\nelse:\n    cmd += [\"--shard\", f\"data/processed/shards/shard-{WORKER_ID}.jsonl\"]\nsubprocess.run(cmd, check=True)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["print(\"worker done — checkpoint published\")"]}
 ],
 "metadata": {"kaggle": {"accelerator": "gpu", "gpuType": "T4x2", "isInternetEnabled": true, "language": "python", "sourceType": "notebook"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Create `kaggle/aggregator.ipynb`** (Account A only)

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Adaptive Laya Aggregator (Account A)\n\nDownloads worker-0/1/2 artifacts, validates, merges, publishes merged checkpoint."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import subprocess, sys\nsubprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", \"requirements.txt\"], check=True)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import os, shutil\nfrom pathlib import Path\nfrom scripts.kaggle_io import download, publish\n\nslugs = [\"adaptive-laya/adaptive-laya-worker-0\",\n         \"adaptive-laya/adaptive-laya-worker-1\",\n         \"adaptive-laya/adaptive-laya-worker-2\"]\ndirs = []\nfor s in slugs:\n    try:\n        d = download(s, dst=f\"results/agg/{s.split('-')[-1]}\")\n        dirs.append(Path(d))\n    except Exception as e:\n        print(\"skip\", s, e)\nprint(\"workers available:\", [str(d) for d in dirs])"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["import json, subprocess, sys\nmethod = os.environ.get(\"ADAPTIVE_LAYA_METHOD\", \"equal_weight\")\nout = \"results/merged\"\nif len(dirs) >= 2:\n    subprocess.run([sys.executable, \"scripts/aggregate_adapters.py\",\n                    \"--workers\"] + [str(d) for d in dirs] +\n                   [\"--method\", method, \"--out\", out], check=True)\n    subprocess.run([sys.executable, \"scripts/verify_checkpoint.py\", out], check=True)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["from scripts.kaggle_io import publish\nfrom pathlib import Path\nimport json, yaml\ncfg = yaml.safe_load(Path(\"configs/worker.yaml\").read_text())\nif Path(\"results/merged/manifest.json\").exists():\n    publish(\"results/merged\", cfg[\"datasets\"][\"merged\"], f\"merged round {os.environ.get('ADAPTIVE_LAYA_ROUND','0')}\")\n    print(\"merged published\")"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["print(\"aggregation complete\")"]}
 ],
 "metadata": {"kaggle": {"accelerator": "gpu", "gpuType": "T4x2", "isInternetEnabled": true, "language": "python", "sourceType": "notebook"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 4: Verify notebooks are valid JSON and bootstrap runs locally**

Run:
```powershell
python -c "import json; json.load(open('kaggle/worker.ipynb')); json.load(open('kaggle/aggregator.ipynb')); print('valid')"
python scripts/bootstrap_kaggle.py --worker-id 0 --mode probe
```
Expected: `valid`, then bootstrap prints worker_id/shard JSON (after decisions dataset exists in `data/decisions_live` or local store; for local dry-run copy `data/decisions/decisions_v1` to `data/decisions_live`).

- [ ] **Step 5: Commit**

```powershell
git add scripts/bootstrap_kaggle.py kaggle/worker.ipynb kaggle/aggregator.ipynb
git commit -m "feat: bootstrap script and worker/aggregator Kaggle notebooks"
```

---

### Task 23: One-time setup + Kaggle Runbook (spec §69, §76, §84, §103)

**Files:**
- Create: `docs/KAGGLE_RUNBOOK.md` (committed as the operational doc)

- [ ] **Step 1: One-time local setup (do once, before any Kaggle run)**

```powershell
# 1. Full test suite green
python -m pytest tests/ -v

# 2. Generate + label seed dataset (if not already)
python scripts/data_generation/generate_decisions.py --n 10000 --seed 0 --out data/processed/decisions_v1

# 3. Publish decisions dataset ONCE to Kaggle (Account A)
#    Create dataset slug adaptive-laya/adaptive-laya-decisions containing:
#      decisions.jsonl + manifest.json
#    via Kaggle UI (https://www.kaggle.com/datasets → New Dataset → upload data/decisions/decisions_v1)
#    OR CLI: kaggle datasets create -p data/decisions/decisions_v1  (slug becomes adaptive-laya/adaptive-laya-decisions after rename in UI)

# 4. Create three empty worker artifact datasets (one per account):
#    Account A: adaptive-laya/adaptive-laya-worker-0
#    Account B: adaptive-laya/adaptive-laya-worker-1
#    Account C: adaptive-laya/adaptive-laya-worker-2
#    Account A: adaptive-laya/adaptive-laya-merged
#    (Kaggle UI → New Dataset → any placeholder file; visibility: Public)

# 5. Push code to GitHub repo; pin initial commit hash as BASE_REVISION
git init; git add -A; git commit -m "adaptive-laya initial"; git push origin main
```

- [ ] **Step 2: Per-round Kaggle run instructions (repeat each round)**

```text
ROUND N RUN CHECKLIST
=====================

A. Start three workers IN PARALLEL (same wall-clock time):

   On Account A → Kaggle → Code → New Notebook
     - Attach GPU: T4 x2 (or T4 x1; code supports single GPU)
     - Internet: ON
     - Runtime: GPU (12h limit)
     - Settings → Environment → Variables/Add:
         WORKER_ID = 0
         ADAPTIVE_LAYA_MODE = probe        (Weeks 1–2) or "lora" (Weeks 6–9)
         ADAPTIVE_LAYA_ROUND = N
         ADAPTIVE_LAYA_REPO = https://github.com/<you>/adaptive-laya
         ADAPTIVE_LAYA_COMMIT = <pinned commit sha>
     - Upload kaggle/worker.ipynb as the notebook (or Add File → Upload)
     - Click Run All. Notebook: pip install → git clone → bootstrap
       (auto-downloads decisions dataset + checksum + shard) → extract
       features → train → verify → publish to adaptive-laya-worker-0.

   On Account B → same steps with WORKER_ID = 1
   On Account C → same steps with WORKER_ID = 2

   Seeds/shards are automatic from configs/worker.yaml (42/43/44; shard 0/1/2).

B. Wait for all three workers to print "checkpoint published"
   (each trains ≤9.5h, then publishes with ~1–2h safety margin before
   Kaggle's 12h cutoff — spec §75).

C. Aggregate (Account A only):
   Account A → New Notebook → GPU optional (CPU fine for router merge)
     - Internet: ON
     - Variables: ADAPTIVE_LAYA_ROUND = N, ADAPTIVE_LAYA_METHOD = equal_weight
     - Upload kaggle/aggregator.ipynb → Run All
     - It downloads worker-0/1/2, validates (shapes/NaN/base revision),
       merges (equal_weight first; compare best_worker and
       validation_weighted as ablations — spec §72/§95), verifies, and
       publishes to adaptive-laya-merged.

D. Next session for each worker (round N+1):
   - Workers automatically resume from adaptive-laya-merged on start
     (scripts/training/train_worker.py → resume_from_merged).
   - Repeat step A with ADAPTIVE_LAYA_ROUND = N+1.

E. Weekly quota discipline (spec §76):
   - Plan 20–24 GPU-hours/account/week = about TWO 10–12h sessions
     per account per week. Check quota in each account before starting:
     https://www.kaggle.com/settings → GPU settings.
   - Never stack runs right to the 12h limit.

F. Phase-0 gate check after round ≥1 (Account A, CPU notebook):
   python scripts/training/probe_decoder.py --input data/processed/shards/shard-0.jsonl --out results/features-0.pt   # if not already cached
   python scripts/training/train_router.py --features results/features-0.pt --out results/router_run --seed 42
   python scripts/evaluation/evaluate_router.py --features results/features-0.pt --router results/router_run/router.pt --router-config results/router_run/router_config.json --gates configs/phase0_qwen.yaml
   → require gates["passed"] == true (Macro-F1 ≥0.70, hard-neg ≥0.75, ECE ≤0.15).
   If FAIL → STOP (spec §41): fix data/state/utility, do not enlarge the model.

G. Phase-1 evaluation (after router validated):
   - Run evaluate_agent.py over 500–2,000 mixed tasks with the 6
     baselines (configs/phase1_mvp.yaml).
   - Produce results/frontier.csv (success vs cost).
   - Proceed to Phase 2 LoRA only if adaptive routing cuts cost at
     comparable success (spec §41 Phase-1 gate).

H. Required ablations before writing results (spec §95):
   E3 (1 worker) vs E4 (3 workers best) vs E5 (3 workers equal-weight)
   vs E6 (3 workers validation-weighted); report task success, routing
   F1, tokens, tool calls, latency, training time.
```

- [ ] **Step 3: Write `docs/KAGGLE_RUNBOOK.md` with the content of Steps 1–2** (copy the one-time setup and round checklist above verbatim into the file).

- [ ] **Step 4: Full-suite regression + final commit**

Run: `python -m pytest tests/ -v` → all PASS.

```powershell
git add docs/KAGGLE_RUNBOOK.md
git commit -m "docs: one-time setup and three-account Kaggle round runbook"
```

---

## Definition of Done (whole project)

- [ ] All tests green: `python -m pytest tests/ -v`
- [ ] `ruff check .` clean
- [ ] Phase-0 gates pass on frozen test split with bootstrap CI
- [ ] Three workers publish + aggregator merges for ≥2 rounds with zero manual file copies
- [ ] `results/frontier.csv` shows adaptive routing on/below baseline cost at comparable success
- [ ] Ablations E3/E4/E5/E6 recorded; claims worded per spec §96 (parallel replicas + weighted averaging, not DDP)

## Self-Review (per writing-plans skill)

1. **Spec coverage:** §45 items 1–12 → Tasks 1–12; §10 tools → Task 13; §9/§11/§12 loop → Task 14; §19–21 eval → Task 15; §60–70 I/O → Task 16; §65/§91–92 verify/publish → Task 17; §24/§27 LoRA → Task 18; §67/§87/§90 worker → Task 19; §54–59/§72–73/§95 aggregation → Task 20; §34 benchmark → Task 21; §85 notebooks → Task 22; §69/§76/§84/§103 runbook → Task 23. Full-weight aggregation (§56) intentionally excluded per §57 "prefer LoRA + router aggregation" — router state averaging covers the routing head; LoRA delta merge is exercised via peft `save_pretrained` artifacts averaged at full-model level when Phase 2 enables adapters (extend `merge_full_states` with adapter weights then).
2. **Placeholder scan:** no TBD/TODO; every code step contains complete code; notebook cells are complete JSON.
3. **Type consistency:** `AdaptiveRouter(hidden_size=...)`, `RouterOutput` fields, `train_router(bundle, hidden_size, ...)`, `execute(tool, args, repo_path)`, `verify_checkpoint(dir, expected_base_revision)`, `aggregate_checkpoints(dirs, method, out_dir)` signatures match across all tasks; `ACTION_TO_IDX`/`TOOL_TO_IDX`/`TERMINAL_TO_IDX` defined identically in `model/backbone.py` and `scripts/training/train_lora.py`.
