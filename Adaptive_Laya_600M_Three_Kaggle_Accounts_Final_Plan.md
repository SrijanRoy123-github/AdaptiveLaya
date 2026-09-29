# Adaptive Laya — Final Kaggle-Only Research Plan

**Version:** Kaggle-Only v2.0 — 600M Generation + Decision Architecture  
**Compute:** Kaggle only; zero paid/external compute  
**Data:** Public/openly licensed resources only, with license verification  
**Primary domain:** Coding agents  
**Primary backbone:** Small pretrained decoder-only model, selected to keep the complete Adaptive Laya model near 600M parameters  
**Core idea:** Learn what computation is sufficient before spending that computation.

## 1. Final Objective

Build a compact coding agent that can:

- generate code and explanations;
- decide whether to answer directly or use a tool;
- choose the appropriate coding tool;
- ask for clarification when essential information is missing;
- re-evaluate after tool/test observations;
- stop when the task is complete;
- operate under explicit token/tool/latency budgets.

The main research claim is **not** frontier-model superiority. The target is a better **quality–compute frontier**: equal or higher task success with fewer unnecessary tokens, tool calls, and wall-clock cost.

```text
User / Repository State
          ↓
    Decoder-only LLM
          ↓
       <|route|>
          ↓
    Adaptive Router
          ↓
   ┌──────┼───────┐
   ↓      ↓       ↓
ANSWER  TOOL     ASK
   │      │       │
   ↓      ↓       ↓
Generate Execute Clarify
   │      │
   └──┬───┘
      ↓
   Observe
      ↓
  Route Again
      ↓
     DONE
```

## 2. What Happens to Laya

The original Laya/mmBERT design becomes a **Phase-0 diagnostic baseline and MVP router**, not the long-term production backbone.

The final architecture uses the core Laya idea—explicit, calibrated decision making—but places the routing head on a decoder-only generative backbone.

The project should compare:

1. **Frozen mmBERT + routing head** — diagnostic probe.
2. **Frozen Qwen code model + routing head** — decoder representation probe.
3. **Qwen code model + learned routing head** — primary integrated architecture.

## 3. Backbone

### Primary candidate

**Qwen2.5-Coder-1.5B-Instruct**.

Its current model card lists Apache-2.0 licensing and describes a 1.54B-parameter causal code model with 28 layers, GQA, tied embeddings, and 32K context. It is designed for code generation, reasoning, fixing, and code-agent use. 

Reference: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct

### Fallback

- Qwen2.5-Coder-0.5B
- SmolLM2-360M

### Rule

**Never train the generator from scratch** under the Kaggle-only constraint.

Use pretrained weights plus LoRA/QLoRA only when generator adaptation is justified.

## 4. Minimal Action Space

Primary actions:

```text
ANSWER
TOOL
ASK
```

Terminal state:

```text
CONTINUE
DONE
ESCALATE
```

Coding tools:

```text
RUN_TESTS
LINT
READ_CONTEXT
SEARCH_REPOSITORY
```

Do not create a large tool/action taxonomy in the MVP.

## 5. Router Architecture

```text
h_route
   ↓
LayerNorm
   ↓
MLP: H → 512 → 256
   ↓
┌─────────┬────────┬────────────┬──────────┐
↓         ↓        ↓            ↓
Action   Tool   Confidence   Terminal
head     head      head          head
```

The router should be small. Target roughly **1–5M trainable parameters** for the first experiments.

Tool prediction is masked unless `action == TOOL`.

## 6. Policy State

The router must use state, not only the original query.

```text
state =
    user_request
  + repository_summary
  + relevant_file_context
  + recent_changes
  + previous_actions
  + test_results
  + previous_tool_result
  + remaining_token_budget
  + remaining_tool_budget
  + remaining_latency_budget
  + step_index
  + previous_confidence
```

Required file:

```text
controller/state_serializer.py
```

Initial compact state target:

```text
256–512 tokens
```

## 7. Budget Conditioning

Use a continuous budget vector rather than verbose context strings:

```text
b = [
  remaining_tokens / max_tokens,
  remaining_tools / max_tools,
  remaining_latency / target_latency,
  step / max_steps
]
```

Project it into the routing hidden dimension and combine it with the route representation.

Suggested initial regimes:

```text
LOW:
  max_tokens = 250
  max_tools = 1

NORMAL:
  max_tokens = 750
  max_tools = 2

HIGH:
  max_tokens = 1500
  max_tools = 4
```

These are experimental settings, not fixed universal values.

## 8. Tool Protocol

Use explicit control tokens:

```text
<|route|>
<|answer|>
<|tool|>
<|ask|>
<|call:run_tests|>
<|call:lint|>
<|call:read_context|>
<|call:search_repo|>
<|args|>
<|args_end|>
<|obs|>
<|obs_end|>
<|done|>
```

Tool selection should come from the classification head. Variable arguments should use grammar/JSON-schema constrained decoding.

## 9. Agent Controller

```text
User
 ↓
Serialize state
 ↓
Run backbone / use KV cache
 ↓
Router
 ↓
ANSWER / TOOL / ASK
 ↓
Execute selected action
 ↓
Format observation
 ↓
Append observation + <|route|>
 ↓
Router again
 ↓
CONTINUE / DONE
```

The controller, not the model weights, owns:

- tool execution;
- permissions;
- timeouts;
- sandboxing;
- context truncation;
- budget accounting;
- logging.

## 10. Sandboxed Coding Tools

### RUN_TESTS
Run selected tests with a strict timeout.

### LINT
Run a deterministic linter.

### READ_CONTEXT
Retrieve relevant repository files/code blocks.

### SEARCH_REPOSITORY
Search symbols, references, imports, functions, and classes.

Do not execute arbitrary generated shell commands directly on the host.

## 11. Context Manager

Raw observations can overflow context.

```text
Tool result
 ↓
truncate
 ↓
deduplicate
 ↓
summarize
 ↓
preserve key errors/results
 ↓
<|obs|> ... <|obs_end|>
```

Initial observation budget:

```text
500–1,000 tokens
```

## 12. Loop Guards

Every task has:

```text
max_steps
max_tool_calls
max_generated_tokens
max_observation_tokens
max_wall_time
```

Example:

```json
{
  "max_steps": 8,
  "max_tool_calls": 3,
  "max_generated_tokens": 1000,
  "max_observation_tokens": 1000
}
```

## 13. Cost Ledger

Router computation must be counted.

```text
Total Cost =
    input_tokens
  + output_tokens
  + α × tool_calls
  + β × router_forwards
  + γ × wall_clock_ms
```

For Architecture A also report:

- Laya inference;
- external generator inference;
- tool time;
- serialization/context-management time.

For Architecture C report:

- decoder prefill;
- token generation;
- router overhead;
- tool time;
- context-management time.

## 14. Utility Function

For a state `s` and action `a`:

```text
U(s,a) =
    success_reward
  - α × generated_tokens
  - β × tool_calls
  - γ × latency
  - δ × router_cost
```

The utility coefficients are hyperparameters and must be logged and ablated.

The label is **not** simply “teacher said this action.”

## 15. Counterfactual Data Generation

For selected tasks, actually execute candidate branches.

Example:

```text
Task: Fix authentication bug

A: Generate patch immediately
   → tests fail

B: Read relevant context
   → patch succeeds

C: Run tests first
   → useful failure information
   → patch succeeds
```

Record:

```text
success
cost
tokens
tool calls
latency
```

Then choose the action with the best measured utility under the current budget.

## 16. Multiple Budget Labels

Each suitable task should be labeled under multiple regimes:

```text
LOW
NORMAL
HIGH
```

This prevents the router from learning a single hardcoded cost tradeoff and makes budget conditioning part of the learned policy.

## 17. Training Data

Use public data and verify licenses before use.

Candidate sources:

### Code generation/fixing

- CommitPackFT
- compatible public code datasets

### Software engineering

- SWE-Gym
- SWE-bench training data

### Unit-test coding

- HumanEval+
- MBPP+
- BigCodeBench

### Tool-delegation ablation

- GSM8K

Do not treat “publicly downloadable” as automatically equivalent to unrestricted commercial use.

Maintain:

```text
data/LICENSES.md
```

with dataset version, source, license, attribution, and allowed use.

## 18. Phase 0 — Decision Probe

### Objective

Test whether routing information is present in frozen representations.

### Probe A

```text
frozen mmBERT
+
router
```

### Probe B

```text
frozen Qwen2.5-Coder
+
router
```

### Dataset

Start with:

```text
10k–20k examples
```

Do not start by manufacturing 50k highly verified examples in a few days.

### Suggested initial training

```text
router parameters only
AdamW
lr = 3e-4 starting point
epochs = 5 starting point
FP16 on T4
```

### Metrics

```text
Macro-F1
TOOL recall
ASK precision
Hard-negative accuracy
ECE
Utility
```

### Baselines

```text
Always ANSWER
Always TOOL
Keyword rules
```

### Development gates

Initial target:

```text
Macro-F1 ≥ 0.70
Hard-negative accuracy ≥ 0.75
ECE ≤ 0.15 after calibration
Positive utility improvement vs best trivial router
```

Keep a frozen test split and use bootstrap confidence intervals.

If the router fails:

```text
STOP
```

Then improve data/state/utility rather than making the model larger.

## 19. Phase 1 — Agent Loop MVP

### Goal

Prove end-to-end value before custom generator work.

Use:

```text
validated router
+
Qwen2.5-Coder-1.5B-Instruct
+
RUN_TESTS
LINT
READ_CONTEXT
SEARCH_REPOSITORY
```

### Evaluation

Use:

```text
500–2,000 mixed tasks
```

with a smaller fixed test subset that is not used to tune thresholds.

### Baselines

```text
Pure generation
Always-answer
Always-test
Keyword router
Naive ReAct-style loop
Adaptive Laya
```

### Primary result

Show the quality–compute frontier:

```text
Y = task success
X = total cost
```

Proceed only if adaptive routing reduces unnecessary computation at comparable success.

## 20. Coding Benchmarks

### Function-level

```text
HumanEval+
MBPP+
```

### Harder coding

```text
BigCodeBench
```

### Agentic

Use a **small, fixed, reproducible SWE-style subset** whose execution cost fits the Kaggle budget.

Do not claim full SWE-bench performance from a tiny subset.

## 21. SWE-style Evaluation

Fix:

```text
repository version
issue version
test command
timeout
subset
seed
```

Record:

```text
successes
failures
timeouts
tokens
tool calls
latency
router forwards
```

## 22. Phase 1 Second-Chance Router

First decision:

```text
issue
 ↓
router
 ↓
read / test / answer
```

Second decision:

```text
observation / draft / failed test
 ↓
router again
 ↓
retry / read / answer / ask / done
```

Only trigger a second decision when justified by:

```text
low confidence
failed tool
failed test
budget condition
```

Compare:

```text
single chance
vs
second chance
```

Measure:

```text
correction rate
additional cost
net utility
```

## 23. ASK Evaluation

If included in the first version, use a scripted user.

Example:

```text
User: Make the function better.

Agent: Which function would you like me to improve?

Scripted user: The Python function shared earlier.
```

Success means:

```text
correct final result after ≤1 clarification
```

Measure:

```text
ASK precision
unnecessary ASK
final task success
```

## 24. Phase 2 — Integrated Generator

Only begin when Phase 1 demonstrates measurable routing value.

Primary architecture:

```text
pretrained decoder-only model
+
<|route|>
+
routing head
+
tool protocol
```

Use LoRA/QLoRA.

Do not train the 1.5B generator from scratch.

## 25. Optional Architecture Comparison

For research purposes only:

### B

```text
mmBERT encoder
+
small causal decoder
```

### C

```text
decoder-only
+
routing head
```

Architecture C is the primary target because it naturally preserves autoregressive KV-cache behavior.

## 26. Phase 2 Training Data

Scale verified trajectories gradually:

```text
5k
→ 10k
→ 25k
```

Only move to the next size when validation shows the previous size has been useful.

## 27. Phase 2 Joint Objective

```text
L =
    L_generation
  + λ_action L_action
  + λ_tool L_tool
  + λ_terminal L_terminal
  + λ_budget L_budget
```

Utility/KL targets can be introduced after the basic supervised router is stable.

## 28. Preference Optimization

Preferred Kaggle-scale path:

```text
candidate trajectories
 ↓
execute
 ↓
verify
 ↓
rank by success + cost
 ↓
rejection sampling / best-of-N
 ↓
SFT
```

Optional:

```text
DPO
```

Do not make PPO a required milestone.

## 29. Adaptive Depth — Future Work

Only after the routed agent is demonstrably useful.

Use **turn-level/sequence-level** early exit first.

Example:

```text
8 layers
16 layers
24 layers
full depth
```

The primary metric must be actual wall-clock latency, not only theoretical FLOPs.

## 30. Sparse MoE — Future Work

Optional only.

Do not add MoE under the initial Kaggle budget.

Only pursue it if measured serving throughput justifies the added complexity.

## 31. Kaggle Compute Plan

Kaggle's current documentation describes 12-hour GPU notebook sessions and 20 GB of auto-saved working storage. Its GPU usage documentation describes a weekly quota of 30 hours or sometimes higher depending on demand/resources. Therefore plan conservatively around **20–24 GPU-hours/week**, reserving the remainder for evaluation, debugging, and failed runs. 

References:
- https://www.kaggle.com/docs/notebooks
- https://www.kaggle.com/docs/efficient-gpu-usage

### Phase 0

```text
Kaggle T4×2
```

Use FP16.

### Phase 1

```text
Kaggle T4×2
```

Use it for controller development and small evaluation runs.

### Phase 2

Still Kaggle-only:

- prefer 4-bit/8-bit loading for inference where supported;
- use LoRA/QLoRA;
- reduce context length;
- checkpoint frequently;
- keep evaluation subsets small.

## 32. Important T4 Rule

Do not assume two T4 cards automatically create one 32 GB VRAM pool.

Treat them as separate GPUs unless the training stack explicitly shards the model.

The first design should be runnable on one T4 if practical, then scale to T4×2 only when necessary.

## 33. Checkpointing

Save every:

```text
500–1,000 steps
```

Persist:

```text
weights
LoRA adapters
optimizer
scheduler
RNG state
step
config
metrics
```

Use a persistent Kaggle Dataset/artifact rather than only ephemeral notebook state.

## 34. Throughput Benchmark

Before a large run:

```text
500–2,000 steps
```

measure:

```text
tokens/sec
samples/sec
steps/sec
VRAM
GPU utilization
wall-clock
```

Then estimate the complete run.

## 35. Evaluation Baselines

Required:

```text
1. Pure generation
2. Always-answer
3. Always-test
4. Keyword/rule router
5. ReAct-style agent
6. Adaptive-RAG-style reference where applicable
7. Laya router + external generator
8. Integrated Adaptive Laya
```

Later:

```text
Oracle router
FLARE-style second-chance baseline
```

## 36. Ablations

Run:

```text
No router
Rule router
Learned router
Accuracy-trained router
Utility-trained router
No budget conditioning
No second chance
No cost penalty
No calibration
```

For the integrated model:

```text
Decoder-only
vs
Encoder-decoder
```

## 37. Statistical Protocol

Use:

```text
bootstrap 95% confidence intervals
```

and at least:

```text
3 random seeds
```

for important router experiments where compute allows.

Do not use the final test set to tune thresholds or training decisions.

## 38. Data Contamination

For benchmark-derived training examples:

- perturb numeric values;
- rename identifiers;
- paraphrase prompts;
- use held-out variants;
- keep benchmark test tasks out of training.

LiveCodeBench or another time-separated benchmark can be added later if the compute budget permits.

## 39. Repository Structure

```text
adaptive-laya/
│
├── README.md
├── LICENSE
│
├── data/
│   ├── README.md
│   ├── LICENSES.md
│   ├── raw/
│   ├── processed/
│   ├── decisions/
│   ├── counterfactuals/
│   └── trajectories/
│
├── configs/
│   ├── phase0_mmbert.yaml
│   ├── phase0_qwen.yaml
│   ├── phase1_mvp.yaml
│   └── phase2_lora.yaml
│
├── model/
│   ├── backbone.py
│   ├── router.py
│   ├── routing_heads.py
│   ├── calibration.py
│   └── losses.py
│
├── controller/
│   ├── agent_loop.py
│   ├── state_serializer.py
│   ├── context_manager.py
│   ├── cost_ledger.py
│   ├── budget_manager.py
│   ├── loop_guard.py
│   └── serialization.py
│
├── tools/
│   ├── registry.py
│   ├── run_tests.py
│   ├── lint.py
│   ├── read_context.py
│   └── search_repo.py
│
├── scripts/
│   ├── data_generation/
│   │   ├── generate_decisions.py
│   │   ├── counterfactuals.py
│   │   └── hard_negatives.py
│   ├── training/
│   │   ├── probe_mmbert.py
│   │   ├── probe_decoder.py
│   │   ├── train_router.py
│   │   └── train_lora.py
│   └── evaluation/
│       ├── evaluate_router.py
│       ├── evaluate_agent.py
│       ├── efficiency.py
│       └── baselines.py
│
└── tests/
    ├── test_router.py
    ├── test_controller.py
    ├── test_serializer.py
    ├── test_tools.py
    └── test_cost_ledger.py
```

## 40. 12-Week Execution Plan

### Weeks 1–2 — Decision Probe

```text
Week 1
- load mmBERT and Qwen Coder
- verify dimensions and latency
- implement state serializer
- implement router
- implement cost ledger
- build seed data

Week 2
- scale to 10k–20k examples
- add hard negatives
- train 3 seeds where feasible
- freeze test split
- evaluate routing + utility + calibration
```

### Weeks 3–5 — Agent MVP

```text
Week 3
- integrate external generator
- implement RUN_TESTS
- implement LINT

Week 4
- implement READ_CONTEXT
- implement SEARCH_REPOSITORY
- context manager
- loop guard
- budget manager

Week 5
- run 500–2,000 tasks
- compare baselines
- generate quality–compute frontier
```

### Weeks 6–9 — Integrated Generator

```text
- warm-start pretrained decoder-only model
- add route token
- add routing head
- LoRA/QLoRA
- tool-call SFT
- verified trajectories
```

### Weeks 10–12 — Optimization and Evaluation

```text
- rejection sampling
- optional DPO
- ablations
- statistical analysis
- frozen final benchmark
- reproducibility package
```

## 41. Final Go / No-Go Rules

### Phase 0

If routing does not beat trivial baselines on held-out utility and hard negatives:

```text
STOP
```

Fix the data/state/utility formulation.

### Phase 1

If adaptive routing does not reduce total cost at comparable task success:

```text
STOP
```

Do not compensate with a bigger model.

### Phase 2

If the integrated decoder does not improve total efficiency over the external-generator system:

```text
Keep Architecture A.
```

The final result does not need to be a single fused model.

## 42. Final Scientific Claim

The project should aim to demonstrate:

> **A compact, budget-conditioned routing policy can dynamically choose among direct generation, coding tools, clarification, and termination, reducing unnecessary computation while maintaining task success under a strict free-compute constraint.**

Do not claim broad superiority over frontier models.

## 43. Final Architecture

```text
                    USER / REPO
                         │
                         ▼
               ┌──────────────────┐
               │ Qwen Coder LLM   │
               │ 0.5B–1.5B        │
               └────────┬─────────┘
                        │
                    <|route|>
                        │
                        ▼
               ┌──────────────────┐
               │ Adaptive Router  │
               │                  │
               │ action           │
               │ tool             │
               │ confidence       │
               │ budget           │
               │ terminal state   │
               └────────┬─────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           ANSWER      TOOL       ASK
              │         │          │
              ▼         ▼          ▼
          GENERATE   EXECUTE    CLARIFY
              │         │
              └────┬────┘
                   ▼
                OBSERVE
                   │
                   ▼
                <|route|>
                   │
                   ▼
                CONTINUE
                   │
                   ▼
                  DONE
```

## 44. Ultimate Principle

```text
Do not always generate.
Do not always use tools.
Do not always use maximum computation.

First decide what is sufficient.
Then spend only the computation that is justified.
```

## 45. Immediate Next Artifact

Before writing the custom generator, implement:

```text
1. state_schema.json
2. action_schema.json
3. tool_schema.json
4. budget_schema.json
5. utility_config.yaml
6. state_serializer.py
7. cost_ledger.py
8. 10k seed decision dataset
9. counterfactual labeler
10. hard-negative generator
11. train_router.py
12. evaluate_router.py
```

This is the final Kaggle-only project boundary. The custom generator, adaptive depth, and sparse experts are all conditional on evidence from the preceding phases.


---

# 46. Three-Kaggle-Account Parallel Training Plan

## Objective

Use three independently authorized Kaggle accounts as three parallel training workers to reduce **elapsed calendar time** and increase data/seed diversity while keeping the project entirely within Kaggle.

Important distinction:

> This is **not true distributed data-parallel training across three Kaggle accounts**. Kaggle notebook sessions do not provide a shared multi-account training process or guaranteed low-latency inter-session communication.

Instead, use:

```text
Account A ──┐
            │
Account B ──┼──> Automated checkpoint exchange ──> Aggregator
            │                                        │
Account C ──┘                                        ▼
                                            Merged Adapter
                                                 │
                         ┌───────────────────────┼──────────────────────┐
                         ▼                       ▼                      ▼
                     Account A              Account B               Account C
```

This is a **federated-style / replica-aggregation workflow**.

### Compliance requirement

Use only separate Kaggle accounts that you legitimately control or are authorized to use, and verify that using multiple accounts in this manner complies with Kaggle's current Terms, quotas, and account rules. Do not create or use accounts solely to evade platform restrictions.

---

# 47. Why Use Three Accounts?

The main benefits are:

```text
1. Parallel wall-clock progress
2. Different data shards
3. Different random seeds
4. Better training-data coverage
5. Independent failure isolation
6. Model averaging / ensemble-like regularization
```

It does **not** reduce total GPU-hours consumed by the training itself.

Example:

```text
1 account:
30 hours of work
→ ~30 elapsed training hours

3 parallel accounts:
30 hours local work each
→ roughly ~30 elapsed hours
→ ~90 aggregate GPU-hours
```

The advantage is **calendar time**, not free extra compute.

---

# 48. Recommended Three-Account Roles

Use stable worker identities:

```text
ACCOUNT-A = Worker 0
ACCOUNT-B = Worker 1
ACCOUNT-C = Worker 2
```

All three should run the same code and model revision.

Their differences should be controlled through configuration.

Example:

```yaml
worker:
  id: 0
  seed: 42
  data_shard: 0
```

```yaml
worker:
  id: 1
  seed: 43
  data_shard: 1
```

```yaml
worker:
  id: 2
  seed: 44
  data_shard: 2
```

---

# 49. Data-Sharding Strategy

Do not give each worker the exact same data.

Partition the training data deterministically:

```text
Full Dataset
    │
    ├── SHARD-0 → Account A
    ├── SHARD-1 → Account B
    └── SHARD-2 → Account C
```

For example:

```text
150,000 examples

Account A:
50,000

Account B:
50,000

Account C:
50,000
```

The split should be based on stable hashes:

```python
shard_id = hash(example_id) % 3
```

This guarantees reproducibility.

---

# 50. Stratified Sharding

Do not simply take:

```text
first 1/3
second 1/3
third 1/3
```

because this can create class imbalance.

Each shard should preserve approximately the same proportions of:

```text
ANSWER
TOOL
ASK
DONE/terminal
difficulty
task family
repository
```

Recommended:

```text
stratified deterministic split
```

---

# 51. Three-Seed Strategy

Use different seeds:

```text
A = 42
B = 43
C = 44
```

This gives:

```text
same base model
same code
different data
different stochasticity
```

The final aggregation therefore combines both:

```text
data diversity
+
optimization diversity
```

---

# 52. Training Synchronization Model

Because the three Kaggle sessions cannot continuously synchronize gradients, use **periodic aggregation rounds**.

Recommended:

```text
Round 0:
same initial model

        ↓

A trains locally
B trains locally
C trains locally

        ↓

Aggregate

        ↓

Merged model

        ↓

A/B/C continue

        ↓

Aggregate again
```

Do not attempt to synchronize every step.

That would require communication infrastructure that Kaggle sessions do not provide reliably.

---

# 53. Recommended Aggregation Frequency

For the first experiment:

```text
2–3 aggregation rounds
```

is enough.

Example:

```text
Round 0
   ↓
5k local steps
   ↓
Aggregate
   ↓
Round 1
   ↓
5k local steps
   ↓
Aggregate
   ↓
Round 2
   ↓
final evaluation
```

The exact number should be chosen from measured throughput.

Too-frequent aggregation increases:

```text
checkpoint publishing
download time
storage
coordination
```

Too-infrequent aggregation increases:

```text
client drift
```

---

# 54. Aggregation Method for LoRA

The recommended training method is:

```text
pretrained base model
+
LoRA adapter
```

Do not duplicate the complete 600M model three times in every aggregation artifact when avoidable.

Each account uploads only:

```text
LoRA adapter weights
+
router weights
+
configuration
+
training metadata
```

The aggregation worker reconstructs the merged model from:

```text
common frozen base
+
merged adapter
+
merged router
```

---

# 55. Weighted LoRA Aggregation

If worker `i` trained on `n_i` examples and produced adapter delta `Δ_i`:

```text
Δ_merged =
Σ (n_i / Σ n_j) × Δ_i
```

For equal shards:

```text
Δ_merged =
(Δ_A + Δ_B + Δ_C) / 3
```

The same principle can be applied to router parameters.

Do not average:

```text
optimizer state
random-number state
scheduler history
```

unless there is a specific experiment for doing so.

---

# 56. Full-Weight Aggregation

If an experiment trains model weights directly rather than LoRA:

```text
W_merged =
Σ (n_i / Σ n_j) × W_i
```

However, full-weight aggregation is more expensive in storage and more sensitive to divergence.

For Kaggle-only Adaptive Laya:

> **Prefer LoRA + router aggregation.**

---

# 57. Better Aggregation: Delta Averaging

A robust representation is:

```text
base_model
      +
adapter_delta
```

Each worker produces:

```text
ΔA
ΔB
ΔC
```

The aggregator computes:

```text
Δ =
weighted_average(ΔA, ΔB, ΔC)
```

Then:

```text
Adaptive Laya checkpoint
=
base_model + Δ
```

This is compact and easy to version.

---

# 58. Aggregation Quality Checks

Never blindly average checkpoints.

After aggregation:

```text
1. Validate tensor names
2. Validate shapes
3. Verify same base model revision
4. Verify same tokenizer
5. Verify same LoRA rank/config
6. Check NaN/Inf
7. Evaluate on validation set
8. Compare against each individual worker
```

Only publish the merged checkpoint if it passes structural validation.

---

# 59. Aggregator Account

Designate:

```text
ACCOUNT-A
```

as the default aggregation worker.

It performs:

```text
download worker-B adapter
download worker-C adapter
load worker-A adapter
merge
evaluate validation subset
publish merged adapter
```

The aggregator can also keep:

```text
best_worker.pt
merged_round_1
merged_round_2
final
```

---

# 60. Automated Artifact Exchange

The complete pipeline should require **no manual dataset/checkpoint uploading** after initial setup.

The preferred exchange path is:

```text
Public data source
      ↓
Each Kaggle notebook downloads automatically

Worker checkpoint
      ↓
Kaggle API / Kaggle Dataset version
      ↓
Aggregator downloads automatically
      ↓
Merged checkpoint
      ↓
New Kaggle Dataset/model version
      ↓
All workers download automatically
```

Kaggle officially supports programmatic access through `kagglehub` and the Kaggle API, including programmatic dataset access. citeturn816758search1turn816758search3

---

# 61. Automatic Data Download

The training notebook should begin with a deterministic bootstrap script:

```text
bootstrap.py
    │
    ├── install dependencies
    ├── verify versions
    ├── download dataset
    ├── verify checksum
    ├── create shard
    ├── download current merged adapter
    └── resume training
```

No manual upload should be necessary.

Kaggle notebooks can access external resources when Internet is enabled, and Kaggle documents package installation through the notebook environment. citeturn816758search0turn816758search2

---

# 62. Data Download Sources

Prefer:

```text
Hugging Face Datasets
Kaggle public datasets
GitHub raw/release files
official public dataset URLs
```

The source should be pinned to:

```text
dataset revision
commit hash
release version
checksum
```

Do not use an unversioned "latest" dataset for the final benchmark.

---

# 63. Automatic Download Example

Conceptually:

```python
from datasets import load_dataset

dataset = load_dataset(
    "DATASET_NAME",
    revision="PINNED_REVISION"
)
```

Or through Kaggle's dataset tooling:

```python
import kagglehub

path = kagglehub.dataset_download("owner/dataset")
```

Store:

```text
source
revision
download timestamp
checksum
```

in the run manifest.

---

# 64. Automatic Checkpoint Pull

At the beginning of a training session:

```text
if merged checkpoint exists:
    download it
    load it
else:
    load original pretrained model
```

This allows any Kaggle session to die and restart without losing the experiment.

---

# 65. Automatic Checkpoint Push

Before the Kaggle session ends:

```text
save adapter
save router
save config
save optimizer
save scheduler
save RNG state
save metrics
save manifest
```

Then publish the required artifacts programmatically.

The notebook should make the decision:

```text
checkpoint number
worker ID
round number
validation score
```

part of the artifact name or metadata.

---

# 66. Shared Experiment Registry

Maintain one machine-readable manifest:

```json
{
  "experiment": "adaptive-laya-600m",
  "round": 1,
  "base_model_revision": "...",
  "worker_a": {
    "seed": 42,
    "shard": 0,
    "steps": 5000,
    "validation_score": 0.74
  },
  "worker_b": {
    "seed": 43,
    "shard": 1,
    "steps": 5000,
    "validation_score": 0.75
  },
  "worker_c": {
    "seed": 44,
    "shard": 2,
    "steps": 5000,
    "validation_score": 0.73
  }
}
```

This manifest is itself published as a versioned artifact.

---

# 67. Automatic Worker Behavior

Each worker follows:

```text
START
  ↓
identify WORKER_ID
  ↓
download pinned dataset
  ↓
create deterministic shard
  ↓
download latest merged adapter
  ↓
train local steps
  ↓
save checkpoint
  ↓
publish checkpoint
  ↓
wait for aggregation round
  ↓
download merged checkpoint
  ↓
continue
```

---

# 68. Aggregator Behavior

```text
WAIT
 ↓
worker A checkpoint available?
worker B checkpoint available?
worker C checkpoint available?
 ↓
YES
 ↓
validate all
 ↓
weighted merge
 ↓
run validation
 ↓
publish merged adapter
 ↓
publish manifest
 ↓
START NEXT ROUND
```

---

# 69. No Manual Data Upload Requirement

The user requirement is:

> Data should automatically download inside each Kaggle environment. No manual dataset upload.

Final implementation:

```text
Notebook starts
     ↓
Internet enabled
     ↓
bootstrap script
     ↓
download public dataset automatically
     ↓
verify checksum
     ↓
cache under /kaggle/working
     ↓
train
```

Kaggle's official notebook documentation states that external downloads and package installation are available when Internet is enabled. citeturn816758search0

---

# 70. No Manual Checkpoint Transfer Requirement

Likewise:

```text
Worker A
   ↓
Kaggle API/Dataset version
   ↓
Aggregator
```

and:

```text
Merged adapter
   ↓
Kaggle Dataset/model version
   ↓
Workers B/C/A
```

All of this should be script-driven.

---

# 71. Important Limitation of Three-Account Aggregation

This method is **not equivalent to 3-GPU synchronous gradient averaging**.

With normal DDP:

```text
step 1:
A gradient ─┐
B gradient ─┼─> average immediately
C gradient ─┘
             ↓
         next step
```

With three Kaggle accounts:

```text
A local steps ──┐
B local steps ──┼─> periodic aggregation
C local steps ──┘
                  ↓
             next round
```

Therefore:

```text
wall-clock:
potentially faster

total compute:
higher

optimization:
FedAvg-like, not DDP
```

This distinction must be stated in the paper/report.

---

# 72. How to Improve Aggregation Quality

Run three types of aggregation experiments:

### Experiment A — Best Worker

```text
select highest validation worker
```

### Experiment B — Simple Average

```text
average all three adapters
```

### Experiment C — Validation-Weighted Average

```text
weight ∝ validation utility
```

Then compare.

This prevents the assumption that averaging must always be better.

---

# 73. Recommended First Aggregation Method

Start with:

```text
equal-weight adapter average
```

because all workers should ideally have equal shard sizes.

Then test:

```text
validation-weighted average
```

as an ablation.

If averaging hurts, use the best checkpoint rather than forcing aggregation.

---

# 74. Parallel Training Schedule

For a target local run:

```text
Round 0
────────
A: 0 → 5k steps
B: 0 → 5k steps
C: 0 → 5k steps

Merge

Round 1
────────
A: merged → 5k
B: merged → 5k
C: merged → 5k

Merge

Round 2
────────
A: merged → final
B: merged → final
C: merged → final
```

The actual step count should be based on the measured Kaggle throughput.

---

# 75. Twelve-Hour Session Design

Never schedule a Kaggle run right up to the 12-hour limit.

Use:

```text
~9–10 hours productive training
+
~1 hour checkpointing/publishing
+
~1–2 hours safety margin
```

Kaggle's notebook documentation currently specifies a 12-hour execution limit for CPU/GPU sessions. citeturn816758search0

---

# 76. Three-Account Weekly Budget

Do not assume the full theoretical quota is guaranteed.

Planning target:

```text
Account A:
20–24 GPU-hours/week

Account B:
20–24 GPU-hours/week

Account C:
20–24 GPU-hours/week
```

Total planned:

```text
~60–72 GPU-hours/week
```

This is a planning budget, not a claim about guaranteed quota availability.

The exact available quota should be checked in each account before a run.

---

# 77. What Three Accounts Give You

At ~20–24 planned hours/account/week:

```text
Single account:
~20–24 planned hours

Three accounts:
~60–72 planned aggregate hours
```

Because they run in parallel, elapsed calendar time can approach the single-account time rather than triple it.

However:

```text
total GPU consumption ≈ 3×
```

and aggregation adds overhead.

---

# 78. Data Shard Assignment for Coding

Suggested specialization:

```text
Worker A:
repository repair / SWE-style tasks

Worker B:
code generation / HumanEval / MBPP style

Worker C:
tool-use / debugging / test-driven trajectories
```

However, completely different distributions can cause specialization and poor averaging.

Therefore the preferred design is:

```text
All workers:
70–80% shared mixture

Each worker:
20–30% specialization
```

This provides diversity without making the adapters incompatible.

---

# 79. Recommended Data Mixture

Example:

```text
Worker A:
70% common
30% SWE repair

Worker B:
70% common
30% code generation

Worker C:
70% common
30% debugging/tool trajectories
```

The common data ensures the workers learn the same general task distribution.

---

# 80. Seed + Shard Matrix

| Worker | Seed | Main shard | Specialization |
|---|---:|---|---|
| A | 42 | Shard 0 | SWE repair |
| B | 43 | Shard 1 | Code generation |
| C | 44 | Shard 2 | Debugging/tools |

Keep validation data **identical and never train on it**.

---

# 81. Validation Must Be Shared

All three accounts should evaluate on:

```text
exact same validation set
```

This is necessary to compare worker checkpoints and aggregation fairly.

The validation set must not be sharded differently.

---

# 82. Test Set Must Never Move

The final test set should be:

```text
frozen
not published as training data
not used for checkpoint selection
not used for aggregation weighting
```

Only run it for final evaluation.

---

# 83. Automatic Experiment Dashboard

Save every round:

```text
worker
round
step
loss
routing F1
tool accuracy
task success
tokens
tool calls
latency
GPU hours
```

Then produce a CSV/JSON summary automatically.

Example:

```text
results/
    round_0.json
    round_1.json
    round_2.json
    workers.csv
    merged.csv
```

---

# 84. GitHub as Code Synchronization

Keep source code in a public/private Git repository if allowed.

Each Kaggle session can automatically execute:

```bash
git clone ...
git checkout PINNED_COMMIT
```

This avoids manually uploading code to three notebooks.

The exact repository access method must be compatible with Kaggle account authentication.

---

# 85. Kaggle Notebook Bootstrap

Each worker should contain only a tiny bootstrap cell:

```text
1. install dependencies
2. clone exact repository revision
3. identify worker
4. download dataset
5. download current adapter
6. run training script
7. publish checkpoint
```

Everything else belongs in the repository.

---

# 86. Suggested Scripts

Add:

```text
scripts/
├── bootstrap_kaggle.py
├── download_data.py
├── make_shards.py
├── train_worker.py
├── publish_checkpoint.py
├── aggregate_adapters.py
├── publish_merged.py
└── verify_checkpoint.py
```

---

# 87. Worker Configuration

Example:

```yaml
project:
  name: adaptive-laya-600m
  round: 0

model:
  base_revision: PINNED_REVISION

worker:
  id: 0
  seed: 42
  shard_count: 3

training:
  local_steps: 5000
  micro_batch_size: 1
  gradient_accumulation: 16

budget:
  max_session_hours: 10.0
```

---

# 88. Aggregation Configuration

```yaml
aggregation:
  method: weighted_lora_average
  weight_by: examples
  validation_gate: true
  min_workers: 3
  rounds: 3
```

Optional:

```yaml
aggregation:
  method: validation_weighted
```

---

# 89. Failure Handling

If one account fails:

```text
A ✓
B ✓
C ✗
```

Do not lose the round.

Options:

```text
1. wait for C to resume
2. aggregate A+B if the experiment protocol permits
3. restart C from latest merged checkpoint
```

Record the failure.

Do not silently change the number of workers.

---

# 90. Automatic Resume

When a session starts:

```text
find latest merged round
        ↓
download
        ↓
find worker-local checkpoint
        ↓
resume
```

If there is no worker-local checkpoint:

```text
resume from latest merged adapter
```

---

# 91. Checkpoint Naming

Use deterministic names:

```text
adaptive-laya/
  round-000/
    worker-000/
    worker-001/
    worker-002/

  round-001/
    worker-000/
    worker-001/
    worker-002/

  merged/
    round-001/
    round-002/
```

Include:

```text
base revision
seed
shard
step
```

in metadata.

---

# 92. Automatic Integrity Verification

Before training:

```text
dataset checksum
model checksum
code commit
config hash
```

Before aggregation:

```text
adapter checksum
tensor shapes
base revision
LoRA config
```

Before final evaluation:

```text
model checksum
dataset revision
test set checksum
```

---

# 93. Reproducibility Goal

Another researcher should be able to:

```text
fork repository
 ↓
create 3 permitted Kaggle workers
 ↓
run bootstrap
 ↓
all data downloads automatically
 ↓
training runs
 ↓
checkpoints exchange automatically
 ↓
aggregated model produced
```

No manual:

```text
dataset upload
checkpoint copying
file renaming
```

should be necessary after the initial project setup.

---

# 94. Important Caveat: Aggregation vs. Better Results

Three workers do **not guarantee better results**.

They can help because:

```text
more data exposure
different stochastic paths
more diverse trajectories
```

but averaging can also hurt if workers drift too far apart.

Therefore the experiment must compare:

```text
single worker
vs
three independent workers
vs
three-worker aggregation
```

This is essential.

---

# 95. Required Aggregation Ablation

Run:

```text
A:
one worker

B:
three independent workers, select best

C:
three workers + simple average

D:
three workers + validation-weighted average
```

Measure:

```text
task success
routing quality
tokens
tool calls
latency
training time
```

---

# 96. Main Claim Under Three-Account Training

Do not write:

> "We used a 3-GPU distributed training cluster."

Instead write:

> "We trained three parallel Kaggle replicas on deterministically partitioned data and periodically aggregated their LoRA/router parameters using weighted averaging."

That is technically accurate.

If direct simultaneous communication is later implemented through a Kaggle-supported mechanism, it can be described separately.

---

# 97. Final Kaggle-Only Architecture

```text
                    PUBLIC DATA SOURCES
                           │
                           ▼
                 Automatic Kaggle Download
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ACCOUNT A           ACCOUNT B          ACCOUNT C
   Worker 0            Worker 1           Worker 2
   Seed 42             Seed 43            Seed 44
   Shard 0             Shard 1            Shard 2
        │                  │                  │
        └──────────────┬───┴───┬──────────────┘
                       ▼
              Automated Artifact Bus
                       │
                       ▼
                 Aggregator
                       │
                LoRA + Router Merge
                       │
                       ▼
                Validation Gate
                       │
                       ▼
              Merged Adaptive Laya
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
      Account A    Account B     Account C
          │            │             │
          └────────────┼─────────────┘
                       ▼
                  Next Round
```

---

# 98. Final Training Pipeline

```text
                    ROUND 0

Public Dataset
      ↓
Automatic download
      ↓
Deterministic 3-way shard
      ↓
┌─────────┬─────────┬─────────┐
│ Worker A│ Worker B│ Worker C│
└────┬────┴────┬────┴────┬────┘
     │         │         │
     ▼         ▼         ▼
   LoRA A    LoRA B    LoRA C
     │         │         │
     └────┬────┴────┬────┘
          ▼         ▼
        Aggregator
             ↓
       Validation
             ↓
       Merged LoRA
             ↓

                    ROUND 1
             same process
```

---

# 99. Final Compute Strategy

Use:

```text
Kaggle Account A
Kaggle Account B
Kaggle Account C
```

in parallel.

Plan conservatively:

```text
~20–24 GPU-hours/account/week
~60–72 aggregate planned GPU-hours/week
```

The actual available quota must be checked for each account.

The 12-hour session limit remains a per-session constraint. citeturn816758search0

---

# 100. What This Changes in the Project

The final project now has two independent scaling dimensions:

### Model scaling

```text
Adaptive Laya-360M
Adaptive Laya-500M
Adaptive Laya-600M
```

### Training scaling

```text
1 Kaggle worker
3 Kaggle workers
```

This gives a clean ablation:

```text
Does a larger model help?

Does parallel diverse training help?

Does aggregation help?

Does adaptive routing help?
```

---

# 101. Final Recommended Experiment Matrix

| Experiment | Model | Workers | Aggregation |
|---|---|---:|---|
| E0 | baseline generator | 1 | none |
| E1 | Adaptive Laya router | 1 | none |
| E2 | Adaptive Laya-500M | 1 | none |
| E3 | Adaptive Laya-600M | 1 | none |
| E4 | Adaptive Laya-600M | 3 | none / best |
| E5 | Adaptive Laya-600M | 3 | equal-weight |
| E6 | Adaptive Laya-600M | 3 | validation-weighted |

The main result should compare E3/E5 against the appropriate baseline.

---

# 102. Final Operating Principle

The project should follow:

```text
Public/Open Data
      ↓
Automatic Download
      ↓
Three Parallel Kaggle Workers
      ↓
Local LoRA + Router Training
      ↓
Automated Checkpoint Publication
      ↓
Automated Aggregation
      ↓
Shared Validation
      ↓
Merged Adaptive Laya-600M
      ↓
Next Training Round
      ↓
Final Frozen Test
```

No manual data upload should be part of the normal training loop.

---

# 103. Final Practical Recommendation

Use **three parallel workers**, but do not overcomplicate the first run.

### First 3-account experiment

```text
Base:
pretrained ~0.5B–0.6B causal model

Trainable:
LoRA + Adaptive Router

Workers:
3

Seeds:
42 / 43 / 44

Data:
3 deterministic stratified shards

Local training:
same number of optimizer steps

Aggregation:
equal-weight LoRA + router average

Aggregation rounds:
2

Validation:
shared frozen set

Test:
shared frozen final set
```

If this beats the one-worker model on the quality–compute frontier, continue with more rounds/data. If not, retain the best architecture without assuming that aggregation is beneficial.

---

# 104. Final Project Boundary

The complete Kaggle-only system is:

```text
             ADAPTIVE LAYA-600M
                    │
          ┌─────────┴─────────┐
          │                   │
     Generation          Adaptive Policy
          │                   │
          │             ┌─────┼──────┐
          │             ▼     ▼      ▼
          │          ANSWER  TOOL   ASK
          │                    │
          │                    ▼
          │               Coding Tools
          │                    │
          └──────────┬─────────┘
                     ▼
                  OBSERVE
                     │
                     ▼
                 ROUTE AGAIN
                     │
                     ▼
                    DONE
```

Training:

```text
3 Kaggle accounts
        ↓
parallel local training
        ↓
automated LoRA/router aggregation
        ↓
shared validation
        ↓
next round
```

Data:

```text
public/open sources
        ↓
automatic download inside Kaggle
        ↓
deterministic sharding
```

Compute:

```text
Kaggle only
no external GPU
no paid compute
```

The final claim remains:

> **A compact ~600M-parameter adaptive generative agent can improve the quality–compute frontier of coding-agent tasks through explicit cost-aware routing, while its entire training and evaluation pipeline remains reproducible on free Kaggle compute.**
