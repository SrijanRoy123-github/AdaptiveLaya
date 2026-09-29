# Adaptive Laya — Kaggle Runbook

## One-Time Local Setup (do once, before any Kaggle run)

```powershell
# 1. Full test suite green
python -m pytest tests/ -v

# 2. Generate + label seed dataset (if not already)
python scripts/data_generation/generate_decisions.py --n 10000 --seed 0 --out data/processed/decisions_v1

# 3. Publish decisions dataset ONCE to Kaggle (Account A)
#    Create dataset slug judata1/adaptive-laya-decisions containing:
#      decisions.jsonl + manifest.json
#    via Kaggle UI (https://www.kaggle.com/datasets → New Dataset → upload data/decisions/decisions_v1)
#    OR CLI: kaggle datasets create -p data/decisions/decisions_v1  (slug becomes judata1/adaptive-laya-decisions after rename in UI)

# 4. Worker artifact datasets (two workers; all under your own account):
#    judata1/adaptive-laya-worker-0
#    judata1/adaptive-laya-worker-1
#    judata1/adaptive-laya-merged
#    (auto-created on first publish by kagglehub.dataset_upload — no manual
#    creation needed; or pre-create via Kaggle UI → New Dataset if preferred)

# 5. Push code to GitHub repo; pin initial commit hash as BASE_REVISION
git init; git add -A; git commit -m "adaptive-laya initial"; git push origin main
```

## Per-Round Kaggle Run Instructions (repeat each round)

ROUND N RUN CHECKLIST
=====================

A. Start both workers IN PARALLEL (same wall-clock time):

   On your account → Kaggle → Code → New Notebook
     - Attach GPU: T4 x2 (or T4 x1; code supports single GPU)
     - Internet: ON
     - Runtime: GPU (12h limit)
     - Settings → Environment → Variables/Add:
         WORKER_ID = 0,1
         ADAPTIVE_LAYA_MODE = probe        (Weeks 1–2) or "lora" (Weeks 6–9)
         ADAPTIVE_LAYA_ROUND = N
         ADAPTIVE_LAYA_REPO = https://github.com/<you>/adaptive-laya
         ADAPTIVE_LAYA_COMMIT = <pinned commit sha>
     - Upload kaggle/worker.ipynb as the notebook (or Add File → Upload)
     - Click Run All. Notebook: pip install → git clone → bootstrap
       (auto-downloads decisions dataset + checksum + shard) → extract
       features → train → verify → publish to adaptive-laya-worker-0/1.
      - First printed line shows the parsed WORKER_ID/MODE/ROUND — check it
        matches what you intended before letting the run proceed.
      - Run All ONCE. NEVER re-run the train cell while a run is active —
        duplicate runs freeze the notebook (GPU oversubscription + orphaned
        processes blocked on a dead stdout pipe). The cell now guards itself
        via /proc and refuses with the offending pids; to proceed, stop the
        session (or run `!pkill -9 -f train_worker.py`) and re-run once.
        Train output streams to results/train-<wid>_<MODE>.log and is tailed
        every 2 min (the cell prints the last log line per worker).

   Seeds/shards are automatic from configs/worker.yaml (42/43; shard 0/1 —
   shard 2 is generated but unused; do NOT change the 3-way shard count,
   published features were extracted from that split).

B. Wait for both workers to print "checkpoint published"
   (each trains ≤9.5h, then publishes with ~1–2h safety margin before
   Kaggle's 12h cutoff — spec §75).

C. Aggregate (any CPU notebook):
   - Variables: ADAPTIVE_LAYA_ROUND = N
     (ADAPTIVE_LAYA_METHOD secret optional — default is validation_weighted)
   - Upload kaggle/aggregator.ipynb → Run All
   - It downloads worker-0/1, validates (shapes/NaN/base revision),
     merges (validation_weighted default; equal_weight/best_worker as
     ablations — spec §72/§95), verifies (skips publish if any worker
     dataset is missing), and publishes to adaptive-laya-merged.

D. Next session for each worker (round N+1):
   - LoRA runs get a fresh per-mode output dir, so they automatically
     resume from adaptive-laya-merged on start (merged router load);
     round ≥2 additionally continues from that worker's previous
     round state (scripts/training/train_lora.py resume helpers).
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
   E3 (1 worker) vs E4 (2 workers best) vs E5 (2 workers equal-weight)
   vs E6 (2 workers validation-weighted); report task success, routing
   F1, tokens, tool calls, latency, training time.
