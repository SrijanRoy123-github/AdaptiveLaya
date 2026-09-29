# Adaptive Laya — Part 2: Phase-0 Probe + Phase-1 Agent (Tasks 9–15) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver deterministic stratified sharding, frozen-feature probes, router training with gates, sandboxed coding tools, the agent loop, and Phase-1 quality–compute frontier evaluation.

**Architecture:** Workers shard the seed decision dataset by stable hash within action strata; frozen backbones pre-extract `<|route|>` hidden states to `.pt` bundles; the router trains only on those features (AdamW, 3e-4, 5 epochs). The agent loop owns tool execution, budgets, and loop guards; the router only decides. Phase-1 evaluation scores task success vs total cost across 6 baselines.

**Tech Stack:** Python 3.10+, PyTorch, transformers, pytest. Tests stay offline (tiny random Qwen2 config; no weight downloads).

**Spec:** `Adaptive_Laya_600M_Three_Kaggle_Accounts_Final_Plan.md` §§18–23, 35–37, 49–51.

**Prerequisites:** Part 1 complete (`docs/superpowers/plans/2026-09-24-adaptive-laya-part1-foundations.md`) — schemas, configs, router, losses, backbone, state serializer, cost ledger, budget/loop guard, seed dataset all exist and tests pass.

**Companion plans:** Part 3 (Tasks 16–23): LoRA worker, 3-account aggregation, Kaggle notebooks + runbook.

---

## File Structure (Part 2 scope)

```
adaptive-laya/
├── scripts/
│   ├── make_shards.py
│   ├── training/{probe_decoder,probe_encoder,train_router,train_lora}.py
│   └── evaluation/{baselines,evaluate_router,evaluate_agent,efficiency}.py
├── tools/{registry,run_tests,lint,read_context,search_repo}.py
├── controller/{context_manager,agent_loop}.py
└── tests/{test_shards,test_probe_train,test_evaluate_router,test_tools,test_agent_loop,test_evaluate_agent}.py
```

**Import rule:** `scripts/training` and `scripts/evaluation` import from `model/`, `controller/`; `tools/` is stdlib-only; `controller/agent_loop.py` imports `tools.registry` (one-way). Tests import with repo root on `pythonpath`.

---

### Task 9: Deterministic stratified sharding (spec §49–§51, §78–§80)

**Files:**
- Create: `scripts/make_shards.py`
- Test: `tests/test_shards.py`

- [ ] **Step 1: Write failing test**

```python
from collections import Counter

from scripts.data_generation.generate_decisions import generate_dataset
from scripts.make_shards import shard_of, shard_records, split_shared_holdouts


def test_shard_deterministic():
    assert shard_of("dec-000001", 3) == shard_of("dec-000001", 3)
    assert shard_of("dec-000001", 3) in {0, 1, 2}


def test_stratified_proportions():
    data = generate_dataset(n=900, seed=0)
    shards = shard_records(data, n=3, seed=42)
    for recs in shards:
        actions = Counter(r["label_action"] for r in recs)
        total = len(recs)
        for act in ("ANSWER", "TOOL", "ASK"):
            assert abs(actions[act] / total - 1 / 3) < 0.05


def test_holdouts_never_in_shards():
    data = generate_dataset(n=300, seed=1)
    val, test, train = split_shared_holdouts(data, val_frac=0.1, test_frac=0.1, seed=0)
    train_ids = {r["id"] for r in train}
    assert not train_ids & {r["id"] for r in val}
    assert not train_ids & {r["id"] for r in test}
    assert not {r["id"] for r in val} & {r["id"] for r in test}
    shards = shard_records(train, n=3, seed=42)
    all_sharded = {r["id"] for s in shards for r in s}
    assert all_sharded == train_ids


def test_deterministic_across_runs():
    data = generate_dataset(n=120, seed=2)
    a = shard_records(data, n=3, seed=42)
    b = shard_records(data, n=3, seed=42)
    assert [[r["id"] for r in s] for s in a] == [[r["id"] for r in s] for s in b]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shards.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/make_shards.py`**

```python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def shard_of(example_id: str, n: int) -> int:
    return _stable_hash(example_id) % n


def shard_records(records: list[dict[str, Any]], n: int, seed: int = 42) -> list[list[dict[str, Any]]]:
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


def split_shared_holdouts(
    records: list[dict[str, Any]],
    val_frac: float,
    test_frac: float,
    seed: int = 0,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = sorted(records, key=lambda r: _stable_hash(r["id"]))
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return val, test, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/decisions/decisions_v1/decisions.jsonl")
    ap.add_argument("--out", type=str, default="data/processed/shards")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    val, test, train = split_shared_holdouts(records, args.val_frac, args.test_frac, seed=0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("val", val), ("test", test)):
        path = out / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in split) + "\n", encoding="utf-8")
    shards = shard_records(train, args.n, seed=args.seed)
    for i, shard in enumerate(shards):
        path = out / f"shard-{i}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in shard) + "\n", encoding="utf-8")
    print(f"train={len(train)} val={len(val)} test={len(test)} shards={[len(s) for s in shards]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and real sharding**

Run: `python -m pytest tests/test_shards.py -v` → PASS
Run: `python scripts/make_shards.py --input data/decisions/decisions_v1/decisions.jsonl --out data/processed/shards`
Expected: `train=... val=... test=... shards=[...]` roughly equal thirds.

- [ ] **Step 5: Commit**

```powershell
git add scripts/make_shards.py tests/test_shards.py
git commit -m "feat: deterministic stratified 3-way sharding with shared holdouts"
```

---

### Task 10: Feature extraction probes (spec §18 Probe A/B)

**Files:**
- Create: `scripts/training/probe_decoder.py`, `scripts/training/probe_encoder.py`
- Shared helper lives in `model/backbone.py` (added below); tested via Task 11.

- [ ] **Step 1: Add shared `extract_bundle` helper to `model/backbone.py`** (append to existing file from Part 1)

```python
from __future__ import annotations as _annotations

from typing import Any, Callable

import torch

ACTION_TO_IDX = {"ANSWER": 0, "TOOL": 1, "ASK": 2}
TOOL_TO_IDX = {"RUN_TESTS": 0, "LINT": 1, "READ_CONTEXT": 2, "SEARCH_REPOSITORY": 3}
TERMINAL_TO_IDX = {"CONTINUE": 0, "DONE": 1, "ESCALATE": 2}


def record_text(rec: dict[str, Any]) -> str:
    return rec["user_request"] + "\n" + rec.get("repository_summary", "")


def extract_bundle(
    records: list[dict[str, Any]],
    hidden_fn: Callable[[str], torch.Tensor],
    desc: str = "extract",
) -> dict[str, Any]:
    from tqdm import tqdm

    feats, labels_action, labels_tool, labels_terminal, labels_conf, ids = [], [], [], [], [], []
    for rec in tqdm(records, desc=desc):
        h = hidden_fn(record_text(rec))
        feats.append(h.squeeze(0).cpu().float())
        labels_action.append(ACTION_TO_IDX[rec["label_action"]])
        labels_tool.append(TOOL_TO_IDX.get(rec.get("label_tool") or "RUN_TESTS", -1))
        labels_terminal.append(TERMINAL_TO_IDX[rec.get("label_terminal", "CONTINUE")])
        labels_conf.append(float(rec.get("label_confidence", 0.5)))
        ids.append(rec["id"])
    return {
        "ids": ids,
        "features": torch.stack(feats),
        "action": torch.tensor(labels_action, dtype=torch.long),
        "tool": torch.tensor(labels_tool, dtype=torch.long),
        "terminal": torch.tensor(labels_terminal, dtype=torch.long),
        "confidence": torch.tensor(labels_conf, dtype=torch.float32),
        "hard_negative": torch.tensor([bool(r.get("hard_negative", False)) for r in records]),
    }
```

- [ ] **Step 2: Implement `scripts/training/probe_decoder.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.backbone import FrozenDecoderBackbone, extract_bundle, load_decoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    model, tok = load_decoder(args.model_id, device=args.device, dtype=torch.float16)
    backbone = FrozenDecoderBackbone(model, tok, device=args.device)
    bundle = extract_bundle(records, backbone.route_hidden, desc="extract")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"saved {bundle['features'].shape} to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement `scripts/training/probe_encoder.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.backbone import FrozenEncoderBackbone, extract_bundle, load_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="google-bert/bert-base-multilingual-cased")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    model, tok = load_encoder(args.model_id, device=args.device)
    backbone = FrozenEncoderBackbone(model, tok, device=args.device)
    bundle = extract_bundle(records, backbone.route_hidden, desc="extract-encoder")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"saved {bundle['features'].shape} to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```powershell
git add model/backbone.py scripts/training/probe_decoder.py scripts/training/probe_encoder.py
git commit -m "feat: shared bundle extraction + decoder/encoder probe CLIs"
```

---

### Task 11: Router training script (spec §18, §45 item 11)

**Files:**
- Create: `scripts/training/train_router.py`
- Test: `tests/test_probe_train.py`

- [ ] **Step 1: Write failing test**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from model.backbone import ROUTE_TOKEN, FrozenDecoderBackbone, extract_bundle
from scripts.data_generation.generate_decisions import generate_dataset
from scripts.training.train_router import train_router


def tiny_bundle(n=48):
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    cfg = Qwen2Config(
        vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bb = FrozenDecoderBackbone(model, tok, route_token=ROUTE_TOKEN, device="cpu")
    records = generate_dataset(n=n, seed=0)
    return extract_bundle(records, bb.route_hidden, desc="test-extract"), 32


def _coeffs():
    return {"lambda_action": 1.0, "lambda_tool": 1.0, "lambda_terminal": 1.0,
            "lambda_conf": 0.5, "lambda_budget": 0.0}


def test_train_router_smoke(tmp_path):
    bundle, hidden = tiny_bundle()
    result = train_router(
        bundle, hidden_size=hidden, epochs=3, batch_size=16, lr=1e-3,
        coeffs=_coeffs(), out_dir=tmp_path, seed=0,
    )
    assert result["final_loss"] < result["first_loss"]
    assert (tmp_path / "router.pt").exists()
    assert (tmp_path / "router_config.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_train_router_seed_reproducible(tmp_path):
    bundle, hidden = tiny_bundle()
    r1 = train_router(bundle, hidden, epochs=1, batch_size=16, lr=1e-3,
                      coeffs=_coeffs(), out_dir=tmp_path / "a", seed=7)
    r2 = train_router(bundle, hidden, epochs=1, batch_size=16, lr=1e-3,
                      coeffs=_coeffs(), out_dir=tmp_path / "b", seed=7)
    assert abs(r1["final_loss"] - r2["final_loss"]) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_probe_train.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/training/train_router.py`**

```python
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

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), out_path / "router.pt")
    (out_path / "router_config.json").write_text(
        json.dumps({"hidden_size": hidden_size, "seed": seed, "epochs": epochs,
                    "batch_size": batch_size, "lr": lr}, indent=2)
    )
    metrics = {"first_loss": first_loss, "final_loss": final_loss,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_probe_train.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```powershell
git add scripts/training/train_router.py tests/test_probe_train.py
git commit -m "feat: router probe training with reproducible seeds and checkpointing"
```

---

### Task 12: Router evaluation with gates and baselines (spec §18, §35, §37, §45 item 12)

**Files:**
- Create: `scripts/evaluation/baselines.py`, `scripts/evaluation/evaluate_router.py`
- Test: `tests/test_evaluate_router.py`

- [ ] **Step 1: Write failing test**

```python
import torch

from scripts.evaluation.baselines import always_answer, always_tool, keyword_router
from scripts.evaluation.evaluate_router import bootstrap_ci, evaluate_predictions, run_gates


def make_eval_set(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    bundle = {
        "features": torch.randn(n, 32, generator=g),
        "action": torch.randint(0, 3, (n,), generator=g),
        "tool": torch.randint(0, 4, (n,), generator=g),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate_router.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/evaluation/baselines.py`**

```python
from __future__ import annotations

import re
from typing import Any

import torch

TOOL_KEYWORDS = re.compile(r"\b(run|test|lint|find|search|grep|fix the failing)\b", re.I)
ASK_KEYWORDS = re.compile(r"\b(better|improve|clean|vague|something|stuff)\b", re.I)


def always_answer(bundle: dict[str, Any]) -> dict[str, torch.Tensor]:
    n = bundle["features"].shape[0]
    return {
        "action": torch.zeros(n, dtype=torch.long),
        "tool": torch.full((n,), -1, dtype=torch.long),
        "terminal": torch.zeros(n, dtype=torch.long),
        "confidence": torch.full((n,), 0.5),
    }


def always_tool(bundle: dict[str, Any]) -> dict[str, torch.Tensor]:
    n = bundle["features"].shape[0]
    return {
        "action": torch.ones(n, dtype=torch.long),
        "tool": torch.zeros(n, dtype=torch.long),
        "terminal": torch.zeros(n, dtype=torch.long),
        "confidence": torch.full((n,), 0.5),
    }


def keyword_router(prompts: list[str]) -> dict[str, torch.Tensor]:
    actions, tools = [], []
    for p in prompts:
        if TOOL_KEYWORDS.search(p):
            actions.append(1)
            tools.append(0)
        elif ASK_KEYWORDS.search(p):
            actions.append(2)
            tools.append(-1)
        else:
            actions.append(0)
            tools.append(-1)
    n = len(prompts)
    return {
        "action": torch.tensor(actions, dtype=torch.long),
        "tool": torch.tensor(tools, dtype=torch.long),
        "terminal": torch.zeros(n, dtype=torch.long),
        "confidence": torch.full((n,), 0.5),
    }
```

- [ ] **Step 4: Implement `scripts/evaluation/evaluate_router.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.calibration import expected_calibration_error
from model.router import AdaptiveRouter


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
    y = bundle["action"].numpy()
    p = preds["action"].numpy()
    tp_tool = int(((y == 1) & (p == 1)).sum())
    fn_tool = int(((y == 1) & (p != 1)).sum())
    fp_tool = int(((y != 1) & (p == 1)).sum())
    tp_ask = int(((y == 2) & (p == 2)).sum())
    fp_ask = int(((y != 2) & (p == 2)).sum())
    tool_recall = tp_tool / (tp_tool + fn_tool) if tp_tool + fn_tool else 0.0
    ask_precision = tp_ask / (tp_ask + fp_ask) if tp_ask + fp_ask else 0.0
    conf = preds["confidence"].float()
    correct = torch.tensor(y == p, dtype=torch.float32)
    hn_mask = bundle.get("hard_negative")
    hn_acc = (
        float((p[hn_mask.numpy()] == y[hn_mask.numpy()]).mean())
        if hn_mask is not None and hn_mask.any() else 0.0
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
        "hard_negative_accuracy": metrics["hard_negative_accuracy"] >= gates["hard_negative_accuracy"],
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate_router.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```powershell
git add scripts/evaluation/baselines.py scripts/evaluation/evaluate_router.py tests/test_evaluate_router.py
git commit -m "feat: router evaluation with macro-F1, ECE, hard negatives, bootstrap CI, gates"
```

---

### Task 13: Sandboxed coding tools (spec §10)

**Files:**
- Create: `tools/registry.py`, `tools/run_tests.py`, `tools/lint.py`, `tools/read_context.py`, `tools/search_repo.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write failing test**

```python
from tools.registry import TOOL_NAMES, execute

SAMPLE = "def add(a, b):\n    return a + b\n"
TEST_PASS = "def test_add():\n    from sample import add\n    assert add(1, 2) == 3\n"
TEST_FAIL = "def test_add():\n    from sample import add\n    assert add(1, 2) == 4\n"


def make_repo(tmp_path, test_body):
    (tmp_path / "sample.py").write_text(SAMPLE)
    (tmp_path / "test_sample.py").write_text(test_body)
    return tmp_path


def test_tool_names():
    assert TOOL_NAMES == ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]


def test_run_tests_pass(tmp_path):
    repo = make_repo(tmp_path, TEST_PASS)
    result = execute("RUN_TESTS", {"target": "test_sample.py", "timeout_s": 60}, str(repo))
    assert result["ok"] is True
    assert "passed" in result["output"]


def test_run_tests_fail(tmp_path):
    repo = make_repo(tmp_path, TEST_FAIL)
    result = execute("RUN_TESTS", {"target": "test_sample.py", "timeout_s": 60}, str(repo))
    assert result["ok"] is False


def test_run_tests_timeout(tmp_path):
    (tmp_path / "test_slow.py").write_text("import time\ndef test_slow():\n    time.sleep(30)\n")
    result = execute("RUN_TESTS", {"target": "test_slow.py", "timeout_s": 2}, str(tmp_path))
    assert result["ok"] is False
    assert result.get("timeout") is True


def test_lint(tmp_path):
    (tmp_path / "bad.py").write_text("import os\n")
    result = execute("LINT", {"path": "bad.py"}, str(tmp_path))
    assert "output" in result


def test_read_context(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n" * 10)
    result = execute("READ_CONTEXT", {"path": "a.py"}, str(tmp_path))
    assert result["ok"] is True
    assert "x = 1" in result["output"]


def test_read_context_missing(tmp_path):
    result = execute("READ_CONTEXT", {"path": "nope.py"}, str(tmp_path))
    assert result["ok"] is False


def test_search_repo(tmp_path):
    (tmp_path / "lib.py").write_text("def find_me():\n    pass\n")
    result = execute("SEARCH_REPOSITORY", {"pattern": "find_me"}, str(tmp_path))
    assert result["ok"] is True
    assert "lib.py" in result["output"]


def test_unknown_tool(tmp_path):
    result = execute("SHELL", {}, str(tmp_path))
    assert result["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `tools/run_tests.py`**

```python
from __future__ import annotations

import subprocess
import sys
from typing import Any


def run_tests(repo_path: str, target: str = ".", timeout_s: int = 60) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", target, "-q", "--tb=short"]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout_s
        )
        output = (proc.stdout + proc.stderr)[-4000:]
        return {"ok": proc.returncode == 0, "output": output, "timeout": False}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return {"ok": False, "output": f"timeout after {timeout_s}s\n{out}", "timeout": True}
```

- [ ] **Step 4: Implement `tools/lint.py`**

```python
from __future__ import annotations

import subprocess
import sys
from typing import Any


def lint(repo_path: str, path: str = ".", timeout_s: int = 30) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "ruff", "check", path, "--output-format", "concise"]
    try:
        proc = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout_s)
        output = (proc.stdout + proc.stderr)[-4000:]
        return {"ok": proc.returncode == 0, "output": output or "no issues", "timeout": False}
    except FileNotFoundError:
        return {"ok": False, "output": "ruff not installed", "timeout": False}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"lint timeout after {timeout_s}s", "timeout": True}
```

- [ ] **Step 5: Implement `tools/read_context.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_context(repo_path: str, path: str, max_chars: int = 4000) -> dict[str, Any]:
    full = Path(repo_path) / path
    try:
        resolved = full.resolve()
        if not str(resolved).startswith(str(Path(repo_path).resolve())):
            return {"ok": False, "output": "path escapes repository", "timeout": False}
        text = resolved.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n...[truncated]..."
        return {"ok": True, "output": text, "timeout": False, "truncated": truncated}
    except FileNotFoundError:
        return {"ok": False, "output": f"not found: {path}", "timeout": False}
```

- [ ] **Step 6: Implement `tools/search_repo.py`**

```python
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def search_repository(repo_path: str, pattern: str, max_hits: int = 50) -> dict[str, Any]:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return {"ok": False, "output": f"bad pattern: {exc}", "timeout": False}
    hits = []
    root = Path(repo_path)
    for file in root.rglob("*.py"):
        if ".git" in file.parts or "site-packages" in file.parts:
            continue
        try:
            for i, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    rel = file.relative_to(root)
                    hits.append(f"{rel}:{i}: {line.strip()}")
                    if len(hits) >= max_hits:
                        break
        except OSError:
            continue
        if len(hits) >= max_hits:
            break
    output = "\n".join(hits) if hits else "no matches"
    return {"ok": True, "output": output, "timeout": False}
```

- [ ] **Step 7: Implement `tools/registry.py`**

```python
from __future__ import annotations

from typing import Any

from tools.lint import lint
from tools.read_context import read_context
from tools.run_tests import run_tests
from tools.search_repo import search_repository

TOOL_NAMES = ["RUN_TESTS", "LINT", "READ_CONTEXT", "SEARCH_REPOSITORY"]

_EXECUTORS = {
    "RUN_TESTS": lambda repo, args: run_tests(
        repo, args.get("target", "."), int(args.get("timeout_s", 60))
    ),
    "LINT": lambda repo, args: lint(repo, args.get("path", ".")),
    "READ_CONTEXT": lambda repo, args: read_context(repo, args.get("path", ".")),
    "SEARCH_REPOSITORY": lambda repo, args: search_repository(repo, args.get("pattern", "")),
}


def execute(tool: str, args: dict[str, Any], repo_path: str) -> dict[str, Any]:
    if tool not in _EXECUTORS:
        return {"ok": False, "output": f"unknown tool: {tool}", "timeout": False}
    return _EXECUTORS[tool](repo_path, args or {})
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools.py -v`
Expected: PASS (all).

- [ ] **Step 9: Commit**

```powershell
git add tools tests/test_tools.py
git commit -m "feat: sandboxed RUN_TESTS/LINT/READ_CONTEXT/SEARCH_REPOSITORY tools"
```

---

### Task 14: Context manager and agent loop (spec §9, §11, §12, §22)

**Files:**
- Create: `controller/context_manager.py`, `controller/agent_loop.py`
- Test: `tests/test_agent_loop.py`

- [ ] **Step 1: Write failing test**

```python
from controller.agent_loop import AgentLoop, MockGenerator, ScriptedRouter
from controller.context_manager import format_observation


def test_format_observation_truncates():
    raw = "error line\n" * 5000
    out = format_observation(raw, max_tokens=100)
    assert len(out.split()) <= 102
    assert out.startswith("<|obs|>")
    assert out.endswith("<|obs_end|>")


def test_format_observation_preserves_errors():
    raw = "AssertionError: expected 3 got 4\n" + ("noise\n" * 1000)
    out = format_observation(raw, max_tokens=200)
    assert "AssertionError" in out


def test_agent_loop_terminates_with_done():
    router = ScriptedRouter([
        {"action": "TOOL", "tool": "READ_CONTEXT", "terminal": "CONTINUE", "confidence": 0.9},
        {"action": "ANSWER", "tool": None, "terminal": "DONE", "confidence": 0.9},
    ])
    loop = AgentLoop(
        router=router,
        generator=MockGenerator("fixed the bug"),
        regime={"max_tokens": 750, "max_tools": 2, "max_latency_ms": 60000, "max_steps": 8},
        loop_limits={"max_steps": 8, "max_tool_calls": 3, "max_generated_tokens": 1000,
                     "max_observation_tokens": 1000, "max_wall_time_ms": 60000},
        repo_path=".",
    )
    result = loop.run({"user_request": "fix bug", "repository_summary": "tiny repo"})
    assert result["finished"] is True
    assert result["steps"] == 2
    assert result["final_answer"] == "fixed the bug"
    assert result["cost"]["tool_calls"] >= 1


def test_agent_loop_budget_exhaustion():
    class AlwaysToolRouter:
        def decide(self, state, budget_vec):
            return {"action": "TOOL", "tool": "READ_CONTEXT", "terminal": "CONTINUE", "confidence": 0.9}

    loop = AgentLoop(
        router=AlwaysToolRouter(),
        generator=MockGenerator("x"),
        regime={"max_tokens": 50, "max_tools": 1, "max_latency_ms": 1000, "max_steps": 2},
        loop_limits={"max_steps": 8, "max_tool_calls": 3, "max_generated_tokens": 1000,
                     "max_observation_tokens": 1000, "max_wall_time_ms": 60000},
        repo_path=".",
    )
    result = loop.run({"user_request": "hi", "repository_summary": "r"})
    assert result["finished"] is True
    assert result["stop_reason"] in {"budget", "loop_guard", "max_steps"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_loop.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `controller/context_manager.py`**

```python
from __future__ import annotations


def _truncate_by_words(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def deduplicate_lines(text: str) -> str:
    seen: set[str] = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def format_observation(raw: str, max_tokens: int = 800) -> str:
    text = deduplicate_lines(raw)
    text = _truncate_by_words(text, max(1, max_tokens - 4))
    return f"<|obs|> {text} <|obs_end|>"
```

- [ ] **Step 4: Implement `controller/agent_loop.py`**

```python
from __future__ import annotations

import time
from typing import Any, Protocol

from controller.budget_manager import BudgetManager
from controller.context_manager import format_observation
from controller.cost_ledger import CostLedger
from controller.loop_guard import LoopGuard
from controller.state_serializer import build_state, serialize_state
from tools.registry import execute


class RouterLike(Protocol):
    def decide(self, state: dict[str, Any], budget_vec: list[float]) -> dict[str, Any]: ...


class GeneratorLike(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, int]: ...


class MockGenerator:
    def __init__(self, text: str = "answer"):
        self.text = text

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, int, int]:
        words = self.text.split()
        return self.text, len(prompt.split()), len(words)


class ScriptedRouter:
    def __init__(self, decisions: list[dict[str, Any]]):
        self.decisions = decisions
        self.i = 0

    def decide(self, state: dict[str, Any], budget_vec: list[float]) -> dict[str, Any]:
        d = self.decisions[min(self.i, len(self.decisions) - 1)]
        self.i += 1
        return dict(d)


class AgentLoop:
    def __init__(
        self,
        router: RouterLike,
        generator: GeneratorLike,
        regime: dict[str, Any],
        loop_limits: dict[str, Any],
        repo_path: str,
        cost_coeffs: dict[str, float] | None = None,
        logger=None,
    ):
        self.router = router
        self.generator = generator
        self.regime = regime
        self.guard = LoopGuard(loop_limits)
        self.repo_path = repo_path
        self.ledger = CostLedger(cost_coeffs or {
            "alpha_tool_calls": 5e-2, "beta_router_forwards": 1e-3, "gamma_wall_ms": 1e-5,
        })
        self.log = logger or (lambda record: None)

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        budget = BudgetManager(self.regime)
        history_actions: list[str] = []
        previous_obs = ""
        test_results = ""
        final_answer = ""
        stop_reason = "max_steps"
        start = time.monotonic()
        step = 0
        generated_tokens = 0
        tool_calls = 0
        obs_tokens = 0

        while True:
            wall_ms = (time.monotonic() - start) * 1000
            if not self.guard.check(step, generated_tokens, tool_calls, obs_tokens, wall_ms):
                stop_reason = self.guard.reason() or "loop_guard"
                break

            state = build_state(
                {**task, "previous_actions": history_actions,
                 "previous_tool_result": previous_obs, "test_results": test_results,
                 "step_index": step, "previous_confidence": 0.5},
                remaining_tokens=budget.remaining_tokens,
                remaining_tools=budget.remaining_tools,
                remaining_latency_ms=budget.remaining_latency_ms,
            )
            state_text = serialize_state(state, self.regime)
            t0 = time.monotonic()
            decision = self.router.decide(state, budget.vector())
            router_ms = (time.monotonic() - t0) * 1000
            self.ledger.add(router_forwards=1, wall_ms=router_ms)

            action = decision["action"]
            history_actions.append(action)
            self.log({"step": step, "decision": decision, "wall_ms": wall_ms})

            if action == "ANSWER" and decision.get("terminal") == "DONE":
                prompt = f"{state_text}\nANSWER:"
                answer, in_tok, out_tok = self.generator.generate(prompt, budget.remaining_tokens)
                final_answer = answer
                self.ledger.add(input_tokens=in_tok, output_tokens=out_tok)
                budget.record_tokens(in_tok + out_tok)
                generated_tokens += out_tok
                stop_reason = "done"
                step += 1
                break

            if action == "TOOL":
                if not budget.can_call_tool():
                    stop_reason = "budget"
                    break
                tool = decision.get("tool") or "READ_CONTEXT"
                t1 = time.monotonic()
                obs = execute(tool, {"path": ".", "pattern": "def ", "target": "."}, self.repo_path)
                tool_ms = (time.monotonic() - t1) * 1000
                self.ledger.add(tool_calls=1, wall_ms=tool_ms)
                budget.record_tool()
                budget.record_latency(tool_ms)
                tool_calls += 1
                raw = obs.get("output", "")
                formatted = format_observation(raw, max_tokens=max(self.regime.get("max_tokens", 750) // 2, 50))
                obs_tokens += len(formatted.split())
                previous_obs = formatted
                if tool == "RUN_TESTS":
                    test_results = raw[-1000:]

            if action == "ASK":
                final_answer = "CLARIFY: " + task.get("user_request", "")
                stop_reason = "asked"
                step += 1
                break

            if budget.exhausted():
                stop_reason = "budget"
                step += 1
                break

            budget.record_step()
            step += 1

        return {
            "finished": True,
            "steps": step,
            "final_answer": final_answer,
            "stop_reason": stop_reason,
            "cost": self.ledger.as_dict(),
            "actions": history_actions,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_loop.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```powershell
git add controller/context_manager.py controller/agent_loop.py tests/test_agent_loop.py
git commit -m "feat: agent loop with context manager, budgets, loop guards, cost ledger"
```

---

### Task 15: Phase-1 agent evaluation and quality–compute frontier (spec §19, §20, §21, §35)

**Files:**
- Create: `scripts/evaluation/evaluate_agent.py`, `scripts/evaluation/efficiency.py`
- Test: `tests/test_evaluate_agent.py`

- [ ] **Step 1: Write failing test**

```python
import json

from scripts.evaluation.efficiency import frontier_point, write_frontier_csv
from scripts.evaluation.evaluate_agent import score_run


def test_score_run_counts_cost():
    run = {
        "finished": True,
        "stop_reason": "done",
        "steps": 3,
        "final_answer": "fixed",
        "cost": {"input_tokens": 100, "output_tokens": 50, "tool_calls": 2,
                 "router_forwards": 3, "wall_ms": 1200, "total_cost": 151.06},
        "actions": ["TOOL", "ANSWER"],
    }
    scored = score_run(run, reference_answer="fixed")
    assert scored["success"] is True
    assert scored["total_cost"] == 151.06
    assert scored["tool_calls"] == 2
    assert scored["router_forwards"] == 3


def test_score_run_failure():
    run = {"finished": True, "stop_reason": "budget", "steps": 2, "final_answer": "wrong",
           "cost": {"input_tokens": 10, "output_tokens": 5, "tool_calls": 0,
                    "router_forwards": 2, "wall_ms": 100, "total_cost": 15.2},
           "actions": ["ANSWER"]}
    scored = score_run(run, reference_answer="fixed")
    assert scored["success"] is False


def test_frontier_point_dominance():
    cheap_ok = frontier_point(success_rate=0.8, total_cost=100.0)
    costly_ok = frontier_point(success_rate=0.8, total_cost=200.0)
    assert cheap_ok["pareto"] is True
    assert costly_ok["pareto"] is False


def test_write_frontier_csv(tmp_path):
    rows = [
        {"system": "always_answer", "success_rate": 0.5, "mean_cost": 80.0, "pareto": True},
        {"system": "adaptive_laya", "success_rate": 0.8, "mean_cost": 90.0, "pareto": True},
    ]
    path = write_frontier_csv(rows, tmp_path / "frontier.csv")
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    header = lines[0].split(",")
    assert header == ["system", "success_rate", "mean_cost", "pareto"]
    json.dumps(rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate_agent.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/evaluation/evaluate_agent.py`**

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score_run(run: dict[str, Any], reference_answer: str) -> dict[str, Any]:
    final = str(run.get("final_answer", "")).strip().lower()
    ref = reference_answer.strip().lower()
    success = bool(run.get("finished")) and final == ref and run.get("stop_reason") == "done"
    cost = run.get("cost", {})
    return {
        "success": success,
        "total_cost": float(cost.get("total_cost", 0.0)),
        "input_tokens": int(cost.get("input_tokens", 0)),
        "output_tokens": int(cost.get("output_tokens", 0)),
        "tool_calls": int(cost.get("tool_calls", 0)),
        "router_forwards": int(cost.get("router_forwards", 0)),
        "wall_ms": float(cost.get("wall_ms", 0.0)),
        "steps": int(run.get("steps", 0)),
        "stop_reason": run.get("stop_reason"),
    }


def aggregate(scored_runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(scored_runs), 1)
    return {
        "n": len(scored_runs),
        "success_rate": sum(r["success"] for r in scored_runs) / n,
        "mean_cost": sum(r["total_cost"] for r in scored_runs) / n,
        "mean_tool_calls": sum(r["tool_calls"] for r in scored_runs) / n,
        "mean_router_forwards": sum(r["router_forwards"] for r in scored_runs) / n,
        "mean_wall_ms": sum(r["wall_ms"] for r in scored_runs) / n,
        "mean_steps": sum(r["steps"] for r in scored_runs) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, required=True, help="JSONL of run results with reference answers")
    ap.add_argument("--out", type=str, default="results/agent_eval.json")
    args = ap.parse_args()

    per_system: dict[str, list[dict[str, Any]]] = {}
    for line in Path(args.runs).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        system = row["system"]
        scored = score_run(row["run"], row["reference_answer"])
        per_system.setdefault(system, []).append(scored)

    summary = {name: aggregate(runs) for name, runs in per_system.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `scripts/evaluation/efficiency.py`**

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def frontier_point(success_rate: float, total_cost: float) -> dict[str, Any]:
    return {"success_rate": success_rate, "mean_cost": total_cost, "pareto": True}


def mark_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: (r["mean_cost"], -r["success_rate"]))
    best_success = -1.0
    for row in sorted_rows:
        if row["success_rate"] > best_success:
            row["pareto"] = True
            best_success = row["success_rate"]
        else:
            row["pareto"] = False
    return rows


def write_frontier_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "success_rate", "mean_cost", "pareto"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in ["system", "success_rate", "mean_cost", "pareto"]})
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True, help="agent_eval.json path")
    ap.add_argument("--out", type=str, default="results/frontier.csv")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = [
        {"system": name, "success_rate": m["success_rate"], "mean_cost": m["mean_cost"]}
        for name, m in summary.items()
    ]
    rows = mark_pareto(rows)
    path = write_frontier_csv(rows, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate_agent.py -v`
Expected: PASS (all).

- [ ] **Step 6: Run full suite and commit**

Run: `python -m pytest tests/ -v` → all PASS.

```powershell
git add scripts/evaluation tests/test_evaluate_agent.py
git commit -m "feat: agent evaluation, baselines aggregation, quality-compute frontier CSV"
```

---

**End of Part 2.** Next: `2026-09-24-adaptive-laya-part3-kaggle-3account.md` (Tasks 16–23 + Kaggle runbook).
