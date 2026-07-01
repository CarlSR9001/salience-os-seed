# SalienceOS Seed

A **non-Transformer**, salience-governed runtime for sequence reasoning. Instead of
a self-attention stack, the core (`core/operators/sass.py`) is a **Salience-Addressable
State Space (SASS)**: stacked state-space blocks with RoPE positions, depthwise
token mixing, and gated recurrence that gives **linear-time, streaming inference**
with persistent per-layer state.

Around that core, a salience signal schedules compute and memory:

- **Runtime orchestrator** (`runtime/orchestrator.py`) coordinates scheduling,
  memory maintenance, adaptive gating, and learning.
- **Controller policy** + **sensor bank** decide where to spend compute.
- **Structured memory** and a **verifier suite** feed back into the runtime.
- An **adaptive coordinator** (weight learner, truth guard) closes a learning loop.
- A `ProtoLanguageModel` produces conversational output from the state-space core.

## What it demonstrates

Non-Transformer reasoning, control, and learning in one runtime — token mixing via
depthwise convolution + gated recurrence (no quadratic attention matrix), persistent
state for truncation-aware streaming, and LoRA-style low-rank adapter hooks for
per-step specialization.

See [`abstract.md`](abstract.md) for the architecture overview (with flowchart) and
[`concept.md`](concept.md) for the SASS-vs-Transformer breakdown.

## Status

Research seed / architecture demonstration — a working exploration of a salience-scheduled
state-space alternative to attention, not a tuned production model. Benchmarks and a
synthetic baseline are under `benchmarks/` and `salience_bench/`.

## Run

```bash
python start.standard.py
```
Requires PyTorch. See `tests/` for component-level checks.
