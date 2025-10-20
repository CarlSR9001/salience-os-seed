# Salience-Addressable State Space (SASS) vs. Transformers

## Core inference loop
`SASSCore` in `core/operators/sass.py` replaces self-attention with a stack of `StateSpaceBlock`s that behave like lightweight state-space filters.

1. Input preparation: Incoming hidden states are RoPE-rotated inside `_apply_rope()` so positions remain encoded without an explicit attention matrix. Hidden dimension must be divisible by four to support the cosine/sine rotation cache built in `_get_rope_factors()`.
2. Block processing: Each `StateSpaceBlock` layer-normalises the sequence, projects it to three tensors (`gates`, `update`, `mix`), and mixes channels via a depthwise 1-D convolution. The gating term blends the new convolutional response with the carried state to produce `new_state`, which is then projected back to model dimension, dropped out, and added residually to the input.
3. State propagation: The block returns both the updated sequence and a detached copy of `new_state`. `SASSCore.forward()` threads these per-layer states across time so later windows can resume from prior context in O(L) time.
4. Hyper-adapter hooks: Optional low-rank deltas supplied via `hyper_deltas` are injected through registered callbacks to permit per-step specialization without expanding the base parameters.
5. Checkpointing: During training the core can route layer execution through `torch.utils.checkpoint` for memory savings when `SASSConfig.use_checkpoint` is enabled.

## Components that mimic Transformer behavior
- Residual connections and layer normalisation surround every block, preserving the deep-network training dynamics used in standard Transformers.
- Rotary positional embeddings (RoPE) provide the same relative position inductive bias popularized in Transformer variants.
- Dropout and linear projections play the role of the Transformer feed-forward stack’s regularisation and channel mixing.
- Low-rank adapter hooks mirror LoRA-style delta injection common in modern Transformer fine-tuning.

## Major deviations from Transformers
- No multi-head attention: Token mixing happens through depthwise convolutions and gated recurrence, avoiding the quadratic attention matrix and achieving linear-time inference.
- Persistent state: Each block keeps a recurrent state tensor that is fed into subsequent calls, enabling streaming and truncation-aware inference that a vanilla Transformer lacks without KV caching.
- Gated convolutional dynamics: Instead of attention scores, the update relies on sigmoid gates and tanh-mixed convolutions, making interactions strictly local unless augmented by teleporters.
- Deterministic channel grouping: There is no query/key/value projection trio; the architecture splits along fixed channel groups governed by `state_channels` and `kernel_size` rather than learned attention heads.
- Salience-driven augmentations: Execution is mediated by the controller in `runtime/orchestrator.py`, which may append SparseJump teleportation or graph reasoning passes based on the salience vector.

## Additional architectural differences in SalienceOS
- Sparse global hops: `SparseJumpTeleporter` in `core/operators/sparse_jump.py` caches selective hidden states per sequence. When salience spikes (e.g., high novelty and alignment), `_execute_action()` blends a retrieved residual into the latest token, approximating attention’s global recall sparingly.
- Controller-mediated scheduling: `SalienceControllerPolicy` (`core/controller/policy.py`) replaces the fixed stack execution loop with a bandit-scored decision over operators. It scores `ControllerAction`s using salience sensors, hysteresis, cooldown, and compute auction bids, only invoking SASS when expected ROI surpasses thresholds.
- S′ scoring engine: `SPrimeController` (`core/controller/s_prime.py`) modulates action scores with novelty, retention, cost penalties, meta-confidence, and yearning dynamics—behaviour absent in Transformer pipelines.
- Sensor-derived inputs: `SensorBank` (`core/sensors/bank.py`) normalises uncertainty, novelty, alignment, progress, cost, drag, and coherence signals that gate both model execution and auxiliary modules.
- Operator tapestry: Within `_execute_action()` the runtime can fall back to memory ops, external tools, verifiers, and reflection loops when salience or budget discourages raw model forward passes.
- Training scaffold: `proto_lm/trainer.py` uses SASS as the language core, yet online updates, salience-triggered training gates, and token birth tracking diverge from standard Transformer pretraining routines.

## Adaptive coordination & persistence

- Training observers: `ProtoLanguageModel.training_step()` now emits structured snapshots (loss components, gradient health, parameter norms) to registered observers. `AdaptiveCoordinator.observe_training()` consumes these snapshots to populate an `AdaptiveVault` and drive downstream analytics.
- Runtime loop integration: Both `ConversationSession.process_user_input()` and `ConversationSession.generate_response()` feed `RuntimeMetrics` into `AdaptiveCoordinator.track_runtime()`. Aggregated salience signals update controller weights through `AdaptiveWeightLearner`, closing the loop between runtime behaviour and policy tuning.
- Truth/elegance/axiom gating: Before returning a reply, `AdaptiveCoordinator.assess_response()` combines salience scores with truth heuristics, elegance tournaments, and axiomatic checks. Gating decisions (e.g., converting risky responses into deferrals) are persisted as `GatingSummary` records.
- Checkpoint persistence: `ConversationSession.save_state()` and `ProtoLanguageModel.register_external_state()` serialize vault contents, gradient flow history, weight learner state, and the most recent gating summary. Reloading a session reconstructs the coordinator and reattaches exporters/importers so online runs resume with identical adaptive context.
- Tests: `tests/test_adaptive_integration.py` exercises coordinator round-trips and session persistence to ensure the adaptive stack survives save/load boundaries.

## Remaining gaps & risks

- No structural growth: Despite tooling for vaulting and elegance judgements, the current model still maintains a fixed parameter count (≈2.64M). Future work must implement tensor resizing or adapter injection paths if autonomous capacity growth is desired.
- Heuristic salience mapping: Novelty/retention/payoff proxies inside `AdaptiveCoordinator.observe_training()` rely on loss ratios rather than learned predictors. Monitoring should verify these heuristics align with observed behaviour and adjust if they introduce bias.
- Safety coverage: Truth and axiom guards operate on scalar salience channels. Extending them to richer evidence (e.g., verifier outputs, chain-of-thought audits) would reduce the chance of false approvals.

## Net effect
SASS preserves Transformer-era conveniences—residual stacks, RoPE positions, adapter friendliness—while trading self-attention for gated state-space dynamics and integrating tightly with a salience-governed controller. The result is a linear-time backbone whose compute allocation, global recall, and specialization are all downstream of real-time salience measurements rather than a static Transformer block schedule.
