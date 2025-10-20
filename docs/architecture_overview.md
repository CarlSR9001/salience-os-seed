# SalienceOS Seed v0.1 – Systems Blueprint

## North-Star Dialectic
Make salience the primitive. Every actor in the system (sensor, controller, operator, verifier) speaks in a shared salience vector space. Compute is doled out proportional to **AIM × KEY × ΔA** while **COST** is always the veto player. The rest of the architecture is plumbing to support that contract.

## Runtime Surfaces
- **`runtime/`**: event loop, orchestration glue, CLI harness.
- **`core/`**: reusable domain-agnostic primitives (sensors, controller, schedulers, operators, memory, meta-state, idea factory).
- **`training/`**: phase-specific routines (pretraining scripts hook into external trainer; calibration datasets; bandit fine-tuning loops).
- **`experiments/`**: small runnable sandboxes validating single subsystems (e.g., salience hysteresis sweep, scheduler interrupts).
- **`assets/`**: configuration defaults, prompt templates, synthetic curricula manifests.

## Module Cartography
### 1. Sensors (`core/sensors/`)
- **`uncertainty.py`**: rolling entropy probes, teacher disagreement estimators.
- **`novelty.py`**: surprisal deltas vs frozen base + n-gram novelty.
- **`alignment.py`**: AIM scoring via cosine between task/goal embeddings.
- **`progress.py`**: KEY predictor (Δ steps remaining estimator) + progress head.
- **`cost.py`**: latency/token cost regressors conditioned on operator + depth.
- **`drag.py`**: friction metrics from structured memory mutation counters.
- **`bank.py`**: orchestrates sampling cadence, MAD normalization, aggregation to the canonical salience vector.

### 2. Salience Controller (`core/controller/`)
- **`policy.py`**: bandit policy with hysteresis + cooldown + compute auction integration.
- **`actions.py`**: enumerations (`cot_depth`, `operator`, `patch`).
- **`trainer.py`**: REINFORCE-style bandit updates fed by verification outcomes.

### 3. Event-Driven Scheduler (`core/scheduler/`)
- **`events.py`**: threshold definitions per sensor.
- **`edn.py`**: event loop gating operator execution; integrates cooldown timers.

### 4. Operators (`core/operators/`)
- **`sass.py`**: state-space core (configurable Mamba-style blocks + teleport KV).
- **`sparse_jump.py`**: rare global hop module.
- **`memory_ops.py`**: verbs for structured tables (`facts`, `hypotheses`, `todos`).
- **`graph_reasoner.py`**: short MPNN bursts with residual fold-back.
- **`patches/`**: dynamic hyper-adapter emitters + optional skill LoRA hooks.
- **`verifier.py`**: math/code calculators, retrieval+NLI, skeptic head.
- **`auction.py`**: compute auction + budget accounting used by controller.

### 5. Meta-State & Self-Awareness (`core/meta/`)
- **`state.py`**: GRU-backed meta-vector maintenance (confidence, blind spots, ROI).
- **`self_report.py`**: textual surface for meta outputs.

### 6. Idea Factory (`core/ideas/`)
- **`generator.py`**: novelty-driven subgoal proposals (MAP-Elites-inspired diversity logic).
- **`simulator.py`**: cheap rollout + ROI scoring.
- **`dispatcher.py`**: push accepted ideas into structured memory `todos`.

### 7. External Memory (`core/memory/`)
- **`tables.py`**: JSON-ish structured stores with transactional edits.
- **`access.py`**: diffable interfaces for operators + salience sensors.

### 8. Runtime Orchestrator (`runtime/orchestrator.py`)
- Step loop pseudocode:
```python
while not done:
    sensors.tick(state, memory, meta)
    salience = sensors.read()
    decision = controller.choose(salience, meta)
    if scheduler.should_fire(salience, decision):
        state = operators.execute(decision, state, memory, patches)
    if verifier.should_run(salience, decision):
        ok, logs = verifier.execute(state, memory)
        controller.learn(salience, decision, ok)
        meta.update(salience, ok, logs)
    if idea_factory.should_propose(salience, meta):
        todos = idea_factory.propose(state, meta, budget)
        memory.todos.enqueue(todos)
    runtime.report(meta)
```
- CLI entrypoint lives in `runtime/cli.py` with YAML/JSON config loader.

## Training Phases Layout (`training/`)
- **`phase0_pretrain.py`**: LM harness (hooks into external trainer, exports ckpt + tokenizer metadata).
- **`phase1_sensors.py`**: supervised calibration datasets + MAD normalization sweep.
- **`phase2_distill.py`**: process distillation ingestion; trace validators.
- **`phase3_bandit.py`**: controller bandit loop with verified reward streams.
- **`phase4_operators.py`**: SparseJump + Graph Reasoner/patch training kits.
- **`phase5_capacity.py`**: MoE graft pipeline + evaluation metrics.

## Configuration Surfaces (`assets/config/`)
- YAML configs keyed by phase/runtime profile (e.g., `runtime_local.yaml`, `bandit_train.yaml`).
- Includes sensor thresholds, cooldown windows, verification budgets, patch gating priors.

## Testing Strategy
- **`tests/`** (PyTest):
  - Sensor calibration invariants (MAD normalization, range checks).
  - Controller hysteresis: flipping only when Δscore > δ across simulated salience sequences.
  - Scheduler interrupts: event firing frequency under synthetic traces.
  - Operators unit loops (SASS forward pass latency, Graph Reasoner convergence).
  - End-to-end smoke test: scripted reasoning task with verification + idea loop.

## Documentation & Notebooks
- `docs/runtime_walkthrough.md`: step-by-step narrative of a runtime episode.
- `docs/training_recipe.md`: expanded instructions for the five-phase curriculum.
- `docs/events_reference.md`: thresholds + expected behaviors.
- `notebooks/`: jupyter experiments (kept lean, targeted for explaining heuristics).

## Immediate Next Steps
1. Flesh out sensor bank + meta-state skeletons with inline documentation.
2. Implement controller + scheduler interplay.
3. Wire operators and runtime orchestrator.
4. Provide runnable smoke tests and config templates.
