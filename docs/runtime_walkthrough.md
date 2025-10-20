# Runtime Walkthrough – SalienceOS Seed v0.1

This document describes a single decision frame inside `SalienceRuntime` and
highlights how each subsystem contributes to salience-native reasoning.

## Step 0 – Inputs
- Runtime receives a `state` mapping with:
  - Latest hidden states / logits / entropy estimates.
  - Structured memory verbs/tool handles (if any).
  - Token accounting metadata (latency, cost).

- `StructuredMemory` tables maintain persistent `facts`, `hypotheses`, and
  `todos`, exposed as a mapping for sensors and operators.

- `MetaState` stores the slowly evolving meta-vector (confidence, difficulty,
  ROI) informed by prior verification outcomes.

## Step 1 – Sense
`SensorBank.tick()` consumes the raw `state`, structured memory snapshot, and
meta snapshot to produce the normalised salience vector. Each scalar is
MAD-normalised, emphasising different control levers:
- **uncertainty** — rolling entropy.
- **novelty (NEW)** — surprisal & n-gram freshness.
- **alignment (AIM)** — goal/prompt cosine.
- **progress (KEY)** — predicted Δsteps remaining.
- **cost** — runtime resource forecast.
- **drag** — tool/memory thrash estimate.

## Step 2 – Decide
`SalienceControllerPolicy` receives salience + meta snapshot and optional auction
bids. It scores every action triple (`cot_depth`, `operator`, `patch`)
according to:
```
Score = [wΔ·ΔA + wA·AIM + wK·KEY + bandit_bias]
         × budget_factor × meta_boost × (1 − k·DRAG)^γ
         − λ·cost  + depth_bonus
```
Hysteresis and cooldown prevent rapid flapping between actions. Bandit weights
(learned from verification reward) bias the policy toward historically
successful actions.

The compute auction (`ComputeAuction`) adds operator-specific bids reflecting
expected ROI after accounting for cost, uncertainty, drag, and remaining budget.

## Step 3 – Schedule
`EventDrivenScheduler` inspects the salience vector and fires events when
thresholds are exceeded (e.g., `novelty_peak`, `uncertainty_spike`). Events gate
expensive operators unless budget is critically low (in which case only
verification proceeds).

## Step 4 – Compute
If the scheduler approves execution, `_execute_action()` routes to the chosen
operator:
- **SASS / SASS+Jump** — passes hidden states through the state-space backbone,
  optionally triggering SparseJump teleporter and Graph-Reasoner when salience
  indicates high AIM × NEW × KEY.
- **MemoryOp** — applies structured verbs to `facts[]`, `hypotheses[]`, or
  `todos[]`.
- **Tool** — runs external calculators/retrievers via tool handles.
- **Verify** — orchestrates verifiers; outcome seeds bandit rewards.

Rewards (progress minus cost penalty, verification success, etc.) update the
bandit to reinforce useful choices.

## Step 5 – Self-awareness & Idea Loop
`MetaState.update()` ingests the salience vector + verification result + budget
status to produce the next meta-vector (clamped to stable ranges).
`IdeaGenerator` monitors salience; when NEW and AIM are high while DRAG is low,
it proposes subgoals. `IdeaSimulator` screens them, and `IdeaDispatcher` enqueues
accepted ideas into `todos[]`.

## Step 6 – Telemetry
`RuntimeMetrics` exposes:
- step index
- chosen decision + hysteresis delta
- meta self-report string
- scheduler snapshot (active events, budget ratio)
- verification outcome (if any)
- accepted idea count this frame

This telemetry feeds dashboards/tests, enabling calibration of salience
thresholds and policy hyperparameters.

## Reflection & Continuous Thought
- **Scratchpad (`core/reflection/scratchpad.py`)** keeps a token-budgeted trail of reasoning traces. `SalienceRuntime` appends REFLECT notes each step, commits traces with salience metadata, and rewards the controller via bandit updates.
- **Pattern library (`core/reflection/patterns.py`)** seeds reusable strategies (e.g., `decompose_complex`). REFLECT actions retrieve patterns matching the current salience profile and log usage stats.
- **Introspection interface (`core/reflection/introspection.py`)** exposes read-only views of salience, controller diagnostics, meta trajectory, memory diffs, verification history, and an optional workspace listing. Runtime updates this interface before each controller decision.
- **Workspace viewer (`core/reflection/workspace.py`)** grants read-only access to a configured root and powers `read`/`scan` tool handlers without risking accidental writes.

## Recursive Language Model (RLM) Scaffold
- Located under `salience_os_seed/rlm/`, the scaffold composes `ModelClient`, `RLMStore`, `RLMPolicy`, and `RLM` orchestrator.
- **Tool contracts** (`rlm/tools.py`) implement `read`, `scan`, `summarize`, `write`, and `spawn` on top of the workspace viewer and scratch storage. Tools enforce JSON schemas and resource limits aligned with the clean-room spec.
- **Policy knobs** (`rlm/policy.py`) set deliberate defaults: `total_budget=40_000`, `per_call_cap=1_536`, `max_depth=4`, `salience_prune_threshold=0.15`, and `confidence_threshold=0.78`. Adjust to tune recursion aggressiveness.
- **Salience-ranked scheduling**: `_materialize_children()` scores proposed child nodes by calling the S′ controller (`core/controller/s_prime.py`) with synthetic salience maps, ensuring high-value branches execute first.
- **Trace & validation**: `tests/test_rlm.py` runs a deterministic `DummyModel` script to confirm `spawn` proposals, salience pruning, and confidence-handling behave as expected.

## Configuration Summary
- **Reflection**: Set `RuntimeConfig.reflection_workspace_root` to enable read-only filesystem introspection for REFLECT actions and RLM tools. Tune `reflection_scratchpad_tokens` and `reflection_history_capacity` to control scratchpad budgets.
- **Sensor calibration**: Enable adaptive sensor probes with `RuntimeConfig.calibration.enabled`. Tune `history_window`/`min_samples` to control how much runtime data each probe ingests, adjust `ridge_penalty` for smoother fits, and balance probe influence with `probe_weight` + `heuristic_regularization` (higher values keep weights closer to the original heuristics).
- **RLM launch**: Instantiate a `ModelClient`, `SalienceRuntime`, and `WorkspaceViewer`, then call `RLM.run(task)` with a tailored `RLMPolicy` for long-context investigations.
