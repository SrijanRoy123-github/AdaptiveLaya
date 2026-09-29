# Kaggle Setup Guide

**One Kaggle account is enough.** The notebook runs both workers in parallel
on the T4 x2 GPUs — a full round finishes in ~1–1.5h for probe, ~9h for LoRA,
well inside one session. Extra accounts are only useful for running multiple
rounds concurrently.

## Before you start
1. Fork/clone this repo to your GitHub account.
2. Pin the commit you want to run (e.g. `feature/adaptive-laya-implementation`).

## Secrets (notebook → Settings gear icon → Add New Secret)

| Secret | Value |
|--------|-------|
| `WORKER_ID` | `0,1` (both T4 GPUs, parallel) or `0` (single worker) |
| `ADAPTIVE_LAYA_MODE` | `probe` or `lora` |
| `ADAPTIVE_LAYA_ROUND` | round number, e.g. `0` |
| `ADAPTIVE_LAYA_REPO` | `https://github.com/<you>/AdaptiveLaya` |
| `GITHUB_TOKEN` | fine-grained PAT with repo read (private repos only) |
| `ADAPTIVE_LAYA_COMMIT` | pinned commit SHA (optional; defaults to latest) |
| `HF_TOKEN` | hf.co/settings → Access Tokens (gated HF datasets) |
| `ADAPTIVE_LAYA_EPOCHS` | probe epochs, default `10` (optional) |
| `ADAPTIVE_LAYA_TEST` | `1` = quick pipeline test (see below), `0`/absent = full run |

## Quick test mode

Set `ADAPTIVE_LAYA_TEST=1` and Run All to validate the whole pipeline in a
few minutes:

- Ingests only `coding,math` (5 small public datasets, ~28 MB), capped at
  600 records/worker into a separate cache (`data/raw/hf_test`)
- 3 small shards (`data/processed/shards_test`), 1 epoch, **no publish**
- Uses separate `_test` output paths, so a later full run is unaffected

Remove the secret (or set `0`) for the real run.

## Datasets (auto-downloaded, all public / non-gated)

| Domain | Source |
|--------|--------|
| coding | `iamtarun/python_code_instructions_18k_alpaca`, `HuggingFaceH4/CodeAlpaca_20K`, `google-research-datasets/mbpp` |
| math | `openai/gsm8k`, `HuggingFaceH4/MATH` |
| reasoning | `Open-Orca/OpenOrca`, `databricks/databricks-dolly-15k` |
| science | `cais/mmlu`, `allenai/sciq` |
| book | `kmfoda/booksum` |
| vlsi | `shailja/Verilog_GitHub` (capped 5k rows) |
| cad | `gnucleus-ai/cad-gen-freecad` |
| circuit | `pphilip/analog-circuits-sky130` (capped 10k rows) |

## Epochs (recommended)

- **Probe/router: 10 epochs** (fast, tiny 0.6M model) — set via `ADAPTIVE_LAYA_EPOCHS`.
- **LoRA: 1 epoch/round × 2 rounds** (`configs/phase2_lora.yaml`). Bump to
  2 epochs/round only if val loss is still dropping; >3 risks overfitting.

## Worker notebook

1. Kaggle → Code → New Notebook (GPU T4 x2, Internet ON)
2. Add the secrets above
3. Upload `kaggle/worker.ipynb`
4. Run All

The notebook clones/pulls the repo, downloads datasets, shards the data,
then runs probe/train as **one subprocess per GPU** (`WORKER_ID=0,1` →
worker 0 on GPU 0, worker 1 on GPU 1). Wall time ≈ single-worker time.

## Aggregator notebook

Optional secrets:
- `ADAPTIVE_LAYA_METHOD` (`validation_weighted` [default] / `equal_weight` / `best_worker`)

Upload `kaggle/aggregator.ipynb`. It refuses to publish a partial merge —
all worker datasets in `configs/worker.yaml` must download successfully.

## Kaggle CLI (optional, local machine)

```powershell
pip install kaggle
# Put ~/.kaggle/kaggle.json: {"username":"...","key":"..."}
kaggle competitions download -c adaptive-laya-600m
```

## GPU limits
- T4 x1 or T4 x2, 9h soft / 12h hard limit per session.
- Never stack runs to the limit — `ADAPTIVE_LAYA_SESSION_HOURS=9.5` auto-stops with safety margin.
