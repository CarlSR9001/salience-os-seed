"""Runtime orchestrator wiring the salience subsystems together."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence

import torch

from ..core.controller import (
    BanditConfig,
    BanditTrainer,
    ControllerAction,
    ControllerDecision,
    ControllerOperator,
    ControllerPatch,
    SalienceControllerPolicy,
)
from ..core.controller.policy import ControllerConfig
from ..core.reflection import (
    IntrospectionInterface,
    PatternLibrary,
    Scratchpad,
    WorkspaceViewer,
)
from ..core.ideas import (
    ExperimentDispatcher,
    IdeaDispatcher,
    IdeaFactoryConfig,
    IdeaGenerator,
    IdeaSimulator,
    SelfExperiment,
)
from ..core.meta import EpisodicStore, MetaState, MetaStateConfig, build_episode, render_self_report
from ..core.memory import (
    MaintenanceThresholds,
    StructuredMemory,
    archive_low_roi_facts,
    merge_redundant_entries,
    prune_failed_hypotheses,
    should_cleanup,
    summarize_old_context,
)
from ..core.operators import (
    AuctionBid,
    ComputeAuction,
    GraphReasoner,
    GraphReasonerConfig,
    MemoryOperator,
    SASSConfig,
    SASSCore,
    SparseJumpTeleporter,
    VerifierSuite,
)
from ..core.sensors import SensorBank
from ..core.scheduler import EventDrivenScheduler, SchedulerConfig


@dataclass
class RuntimeConfig:
    """Top-level runtime configuration (seed release defaults)."""

    budget_tokens: int = 1024
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    controller_bandit: BanditConfig = field(default_factory=BanditConfig)
    meta: MetaStateConfig = field(default_factory=MetaStateConfig)
    graph_reasoner: GraphReasonerConfig = field(default_factory=GraphReasonerConfig)
    sass: SASSConfig = field(default_factory=SASSConfig)
    idea_factory: IdeaFactoryConfig = field(default_factory=IdeaFactoryConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    reflection_workspace_root: Optional[str] = None
    reflection_scratchpad_tokens: int = 512
    reflection_history_capacity: int = 128
    maintenance_enabled: bool = True
    maintenance: MaintenanceThresholds = field(default_factory=MaintenanceThresholds)
    self_experiments_enabled: bool = True
    experiments_parameters: Sequence[str] = field(
        default_factory=lambda: ("controller.lambda_cost", "scheduler.min_budget_ratio")
    )
    experiments_duration: int = 16
    experiments_max_concurrent: int = 1
    episodic_enabled: bool = True
    episodic_store_path: Optional[str] = None


@dataclass
class RuntimeMetrics:
    """Telemetry emitted per step."""

    step: int
    decision: ControllerDecision
    meta_report: str
    verification_passed: Optional[bool]
    budget_left: float
    scheduler_snapshot: Mapping[str, object]
    idea_acceptances: int
    yearning_snapshot: Mapping[str, Dict[str, float]]
    maintenance_report: Mapping[str, int] = field(default_factory=dict)
    experiment_reports: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    episode_recorded: Optional[str] = None
    salience_raw: Mapping[str, float] = field(default_factory=dict)


@dataclass
class _ActiveExperiment:
    experiment: SelfExperiment
    remaining_steps: int
    originals: Dict[str, float]


class SalienceRuntime:
    """Main runtime driver for SalienceOS Seed."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.memory = StructuredMemory()
        self._sensor_memory: Dict[str, object] = {}
        self._verification_history: Deque[int] = deque(maxlen=64)
        self._verification_rate: float = 0.0
        self.sensor_bank = SensorBank.default_bank()
        salience_dim = len(self.sensor_bank.ordering)
        if self.config.meta.salience_dim != salience_dim:
            self.config.meta = replace(self.config.meta, salience_dim=salience_dim)
        self.meta_state = MetaState(self.config.meta)
        self.meta_state.vector[self.config.meta.confidence_index] = 0.5
        self.meta_state.vector[self.config.meta.roi_index] = 0.3
        self.sass = SASSCore(self.config.sass)
        self.teleporter = SparseJumpTeleporter()
        self.graph_reasoner = GraphReasoner(self.config.graph_reasoner)
        self.verifier = VerifierSuite()
        self.idea_generator = IdeaGenerator(self.config.idea_factory)
        self.idea_simulator = IdeaSimulator()
        self.idea_dispatcher = IdeaDispatcher(self.memory)
        self.memory_operator = MemoryOperator(self.memory)
        bandit_store: Dict[str, Dict[str, float]] = {}
        self.bandit_trainer = BanditTrainer(self.config.controller_bandit, bandit_store)
        self.controller = SalienceControllerPolicy(self.config.controller, bandit_store)
        self.scheduler = EventDrivenScheduler(config=self.config.scheduler)
        self.auction = ComputeAuction()
        self.maintenance_thresholds = self.config.maintenance
        self.experiment_dispatcher = (
            ExperimentDispatcher(
                self.config.experiments_parameters,
                max_concurrent=self.config.experiments_max_concurrent,
            )
            if self.config.self_experiments_enabled
            else None
        )
        self._active_experiments: List[_ActiveExperiment] = []
        self.budget_left = float(self.config.budget_tokens)
        self.step_index = 0
        self.layer_states: Optional[list[torch.Tensor]] = None
        self.hidden_states: Optional[torch.Tensor] = None
        self.workspace_viewer = (
            WorkspaceViewer(self.config.reflection_workspace_root)
            if self.config.reflection_workspace_root
            else None
        )
        self.scratchpad = Scratchpad(
            max_tokens=self.config.reflection_scratchpad_tokens,
            history_capacity=self.config.reflection_history_capacity,
        )
        self.pattern_library = PatternLibrary()
        self.introspection = IntrospectionInterface(
            controller=self.controller,
            verifier=self.verifier,
            memory=self.memory,
            meta_state=self.meta_state,
            workspace=self.workspace_viewer,
        )
        self._dynamics_history: list[Dict[str, object]] = []
        episodic_path = None
        if self.config.episodic_enabled:
            if self.config.episodic_store_path:
                episodic_path = Path(self.config.episodic_store_path).expanduser()
        self.episodic_store = EpisodicStore(episodic_path) if self.config.episodic_enabled else None

    def run_step(
        self,
        state: Mapping[str, object],
    ) -> RuntimeMetrics:
        """Execute a single runtime iteration."""

        enriched_state = self._enrich_state_for_sensors(state)
        state = enriched_state
        meta_snapshot = dict(self.meta_state.snapshot())
        meta_snapshot.setdefault("verification_pass_rate", self._verification_rate)
        structured_snapshot = self.memory.as_runtime_mapping()
        self.introspection.attach_memory_snapshot(structured_snapshot)
        sensor_memory = self._prepare_sensor_memory(structured_snapshot)
        salience_vector = self.sensor_bank.tick(state, sensor_memory, meta_snapshot)
        salience_map = salience_vector.as_mapping()
        self.introspection.attach_salience(salience_map)
        self._prepare_auction_bids(salience_map, meta_snapshot)
        decision = self._choose_with_defaults(salience_map, meta_snapshot)
        should_run = self.scheduler.should_fire(
            salience_map,
            decision_operator=decision.action.operator.name,
            budget_left=self.budget_left,
            budget_total=self.config.budget_tokens,
        )
        verification_passed: Optional[bool] = None
        if should_run:
            verification_passed = self._execute_action(decision, state, salience_map)
        accepted_ideas = self._maybe_generate_ideas(salience_map, meta_snapshot)
        self.step_index += 1
        self.budget_left = max(0.0, self.budget_left - state.get("token_cost", 1.0))
        meta_vector = self.meta_state.update(
            salience_vector=salience_map,
            verification_passed=verification_passed,
            budget_left=self.budget_left,
            cooldown_active=self.controller.state.cooldown_remaining > 0,
        )
        self._update_verification_history(verification_passed)
        cleanup_report = self._maybe_run_maintenance(salience_map)
        experiment_reports = self._advance_self_experiments(
            salience_map,
            meta_snapshot,
            verification_passed,
        )
        episode_id = self._record_episode(salience_map, decision, verification_passed)
        yearning_snapshot = self.introspection.get_yearning_state(refresh=True)
        metrics = RuntimeMetrics(
            step=self.step_index,
            decision=decision,
            meta_report=render_self_report(self.meta_state.snapshot()),
            verification_passed=verification_passed,
            budget_left=self.budget_left,
            scheduler_snapshot=self.scheduler.snapshot(),
            idea_acceptances=accepted_ideas,
            yearning_snapshot=yearning_snapshot,
            maintenance_report=cleanup_report,
            experiment_reports=experiment_reports,
            episode_recorded=episode_id,
            salience_raw={reading.name: reading.raw for reading in salience_vector.readings},
        )
        return metrics

    def _prepare_sensor_memory(self, snapshot: Mapping[str, object]) -> Dict[str, object]:
        for key, value in snapshot.items():
            self._sensor_memory[key] = value
        return self._sensor_memory

    def _update_verification_history(self, outcome: Optional[bool]) -> None:
        if outcome is None:
            return
        self._verification_history.append(1 if outcome else 0)
        if self._verification_history:
            self._verification_rate = sum(self._verification_history) / len(self._verification_history)

    def update_sensor_context(self, key: str, value: object) -> None:
        self._sensor_memory[key] = value

    def _choose_with_defaults(
        self,
        salience_map: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
    ) -> ControllerDecision:
        decision = self.controller.choose(salience_map, meta_snapshot, self.budget_left)
        action = decision.action
        cost_pressure = 1.0 - min(0.99, self.budget_left / max(self.config.budget_tokens, 1.0))
        fatigue = float(meta_snapshot.get("fatigue", 0.0))
        allow_train = self._should_allow_training(salience_map, meta_snapshot, cost_pressure, fatigue)
        if action.operator is ControllerOperator.REFLECT:
            return decision
        if action.operator is ControllerOperator.SASS and action.cot_depth < 2:
            patched_action = ControllerAction(cot_depth=2, operator=ControllerOperator.REFLECT, patch=action.patch)
            return ControllerDecision(
                action=patched_action,
                score=decision.score,
                salience_mapping=decision.salience_mapping,
                cooldown_steps=decision.cooldown_steps,
                hysteresis_delta=decision.hysteresis_delta,
            )
        if action.operator not in {ControllerOperator.REFLECT, ControllerOperator.VERIFY} and not allow_train:
            patched_action = ControllerAction(cot_depth=max(action.cot_depth, 2), operator=ControllerOperator.REFLECT, patch=ControllerPatch.NONE)
            return ControllerDecision(
                action=patched_action,
                score=decision.score,
                salience_mapping=decision.salience_mapping,
                cooldown_steps=decision.cooldown_steps,
                hysteresis_delta=decision.hysteresis_delta,
            )
        return decision

    def _should_allow_training(
        self,
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
        cost_pressure: float,
        fatigue: float,
    ) -> bool:
        novelty = float(salience.get("novelty", 0.0))
        retention = float(salience.get("retention", 0.0))
        payoff = float(salience.get("roi", 0.0))
        s_prime_proxy = 0.35 * novelty + 0.35 * retention + 0.30 * payoff
        fatigue_penalty = cost_pressure + fatigue
        return s_prime_proxy >= 0.55 and fatigue_penalty < 0.25

    def _prepare_auction_bids(self, salience: Mapping[str, float], meta_snapshot: Mapping[str, float]) -> None:
        self.auction.clear()
        uncertainty = float(salience.get("uncertainty", 0.0))
        progress = float(salience.get("progress", 0.0))
        cost = float(salience.get("cost", 0.0))
        drag = float(salience.get("drag", 0.0))
        confidence = float(meta_snapshot.get("confidence", 0.0))
        budget_factor = 1.0 + 0.5 * max(0.0, self.budget_left / max(self.config.budget_tokens, 1.0) - 0.5)
        for action in self.controller.actions:
            depth_bonus = 0.25 * action.cot_depth * uncertainty
            operator_bias = 0.0
            if action.operator is ControllerOperator.VERIFY:
                operator_bias = 0.8 * max(progress, 0.0)
            elif action.operator is ControllerOperator.TOOL:
                operator_bias = 0.4 * uncertainty
            elif action.operator is ControllerOperator.MEMORY_OP:
                operator_bias = 0.3 * (1.0 - drag)
            else:  # SASS family
                operator_bias = 0.5 * progress
            expected_gain = max(0.0, operator_bias + depth_bonus) * budget_factor
            expected_cost = self.config.controller.lambda_cost * cost * (1.0 - 0.2 * confidence)
            self.auction.submit(AuctionBid(action=action, expected_gain=expected_gain, expected_cost=expected_cost))
        for action, score in self.auction.resolve().items():
            self.controller.register_auction_bid(action, score)
        self.auction.clear()

    def _execute_action(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:
        action = decision.action
        operator = action.operator
        if operator is ControllerOperator.VERIFY:
            context = dict(state)
            context.setdefault("memory_snapshot", self.memory.as_runtime_mapping())
            outcome = self.verifier.run(context)
            reward = (1.0 if outcome.passed else -1.0) - self.config.controller.lambda_cost * float(salience.get("cost", 0.0))
            self.bandit_trainer.update(action, reward)
            self.introspection.attach_verification(outcome)
            return outcome.passed

        if operator in {ControllerOperator.SASS, ControllerOperator.SASS_WITH_JUMP}:
            hidden_states = state.get("hidden_states")
            if hidden_states is None:
                if self.hidden_states is None:
                    raise ValueError("SASS execution requires `hidden_states` in state or cached hidden state")
                hidden_states = self.hidden_states
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("`hidden_states` must be a torch.Tensor")
            layer_states = state.get("layer_states") or self.layer_states
            hyper_deltas = state.get("hyper_deltas")
            output, new_layer_states = self.sass(hidden_states, layer_states, hyper_deltas)
            sequence_id = int(state.get("sequence_id", 0))
            if operator is ControllerOperator.SASS_WITH_JUMP:
                trigger = bool(
                    state.get("teleport_trigger")
                    or (
                        float(salience.get("novelty", 0.0)) > 1.0
                        and float(salience.get("alignment", 0.0)) > 0.8
                    )
                )
                residual = self.teleporter(output, sequence_id=sequence_id, trigger=trigger)
                output[:, -1, :] = output[:, -1, :] + residual
                if float(salience.get("progress", 0.0)) > 0.8 or state.get("reasoner_trigger"):
                    output, _ = self.graph_reasoner(
                        output,
                        self.memory,
                        entity_hints=state.get("entity_hints", ()),
                    )
            self.layer_states = new_layer_states
            self.hidden_states = output.detach()
            reward = float(salience.get("progress", 0.0)) - self.config.controller.lambda_cost * float(salience.get("cost", 0.0))
            self.bandit_trainer.update(action, reward)
            return None

    def adjust_controller_dynamics(self, updates: Mapping[str, float], *, max_step: float = 0.1) -> Dict[str, float]:
        applied = self.introspection.adjust_controller_dynamics(updates, max_step=max_step)
        if applied:
            self._dynamics_history.append({
                "step": self.step_index,
                "applied": dict(applied),
            })
            if len(self._dynamics_history) > 64:
                self._dynamics_history.pop(0)
        return applied

    def dynamics_history(self) -> Sequence[Dict[str, object]]:
        return list(self._dynamics_history)

        if operator is ControllerOperator.MEMORY_OP:
            verb = state.get("memory_verb")
            if isinstance(verb, Mapping):
                result = self.memory_operator.execute(verb)
                reward = (1.0 if result.applied else -0.2) - 0.1 * float(salience.get("drag", 0.0))
                self.bandit_trainer.update(action, reward)
            return None

        if operator is ControllerOperator.TOOL:
            tools = state.get("tools")
            tool_name = state.get("tool_name") or action.patch.name.lower()
            if isinstance(tools, Mapping) and tool_name in tools:
                tool_fn = tools[tool_name]
                if callable(tool_fn):
                    tool_fn(state)
                    self.bandit_trainer.update(action, 0.5)
            return None

        if operator is ControllerOperator.REFLECT:
            self._perform_reflection(action, state, salience)
            return None

        return None

    def _maybe_generate_ideas(self, salience: Mapping[str, float], meta_snapshot: Mapping[str, float]) -> int:
        if not self.idea_generator.should_generate(salience):
            return 0
        proposals = self.idea_generator.generate(salience, meta_snapshot, self.memory)
        if not proposals:
            return 0
        simulations = self.idea_simulator.simulate(proposals, meta_snapshot, salience)
        accepted = self.idea_dispatcher.dispatch(simulations, meta_snapshot)
        return len(accepted)

    def _maybe_run_maintenance(self, salience: Mapping[str, float]) -> Dict[str, int]:
        if not self.config.maintenance_enabled:
            return {}
        thresholds = self.maintenance_thresholds
        if not should_cleanup(salience, self.memory, thresholds):
            return {}
        report: Dict[str, int] = {}
        archived = archive_low_roi_facts(self.memory, threshold=thresholds.low_roi_threshold)
        if archived:
            report["archived"] = archived
        merged = merge_redundant_entries(self.memory, thresholds.merge_similarity)
        if merged:
            report["merged"] = merged
        pruned = prune_failed_hypotheses(self.memory, thresholds.verification_failure_limit)
        if pruned:
            report["pruned"] = pruned
        summarized = summarize_old_context(
            self.memory,
            age_threshold=thresholds.summary_age_threshold,
            chunk_size=thresholds.summary_chunk,
        )
        if summarized:
            report["summarized"] = summarized
        return report

    def _advance_self_experiments(
        self,
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
        verification_passed: Optional[bool],
    ) -> List[Mapping[str, object]]:
        if not self.experiment_dispatcher:
            return []
        reports: List[Mapping[str, object]] = []
        proposed = self.experiment_dispatcher.propose_experiment(salience, verification_passed)
        if proposed:
            duration = max(1, self.config.experiments_duration)
            proposed.duration_steps = duration
            baseline = 1.0 if verification_passed else 0.0 if verification_passed is False else 0.5
            proposed.baseline_verification = baseline
            originals = self._apply_overrides(proposed.parameter_overrides)
            self._active_experiments.append(
                _ActiveExperiment(experiment=proposed, remaining_steps=duration, originals=originals)
            )
        finished: List[_ActiveExperiment] = []
        for active in self._active_experiments:
            experiment = active.experiment
            verification_value = 1.0 if verification_passed else 0.0 if verification_passed is False else 0.5
            experiment.metrics.setdefault("verification", []).append(verification_value)
            experiment.metrics.setdefault("drag", []).append(float(salience.get("drag", 0.0)))
            experiment.metrics.setdefault("cost", []).append(float(salience.get("cost", 0.0)))
            active.remaining_steps -= 1
            if active.remaining_steps <= 0:
                analysed = self.experiment_dispatcher.analyze_results(experiment)
                reports.append(
                    {
                        "name": analysed.name,
                        "conclusion": analysed.conclusion,
                        "results": analysed.results,
                    }
                )
                self._revert_overrides(active.originals)
                finished.append(active)
        for experiment in finished:
            self._active_experiments.remove(experiment)
        return reports

    def _apply_overrides(self, overrides: Mapping[str, float]) -> Dict[str, float]:
        originals: Dict[str, float] = {}
        for key, value in overrides.items():
            obj, attr = self._resolve_override_target(key)
            originals[key] = getattr(obj, attr)
            setattr(obj, attr, float(value))
        return originals

    def _revert_overrides(self, originals: Mapping[str, float]) -> None:
        for key, value in originals.items():
            obj, attr = self._resolve_override_target(key)
            setattr(obj, attr, float(value))

    def _resolve_override_target(self, key: str) -> tuple[object, str]:
        try:
            prefix, attr = key.split(".", 1)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Invalid parameter override key '{key}'") from exc
        if prefix == "controller":
            return self.controller.config, attr
        if prefix == "scheduler":
            return self.scheduler.config, attr
        if prefix == "meta":
            return self.meta_state.config, attr
        raise KeyError(f"Unsupported parameter override prefix '{prefix}'")

    def _record_episode(
        self,
        salience: Mapping[str, float],
        decision: ControllerDecision,
        verification_passed: Optional[bool],
    ) -> Optional[str]:
        if not self.episodic_store:
            return None
        scratch_summary = self.scratchpad.summarize(max_traces=2)
        lessons = (
            "verification_passed" if verification_passed else "verification_failed" if verification_passed is False else "no_verification"
        )
        episode = build_episode(
            salience_profile=salience,
            actions=[decision.action.operator.name],
            outcome=verification_passed,
            scratchpad_summary=scratch_summary,
            lessons=lessons,
            task_type="runtime_step",
            metadata={
                "step": self.step_index,
                "decision_score": decision.score,
                "cooldown": decision.cooldown_steps,
            },
        )
        recorded = self.episodic_store.record_episode(episode)
        return recorded.episode_id

    def _perform_reflection(
        self,
        action: ControllerAction,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> None:
        sorted_salience: Sequence[tuple[str, float]] = sorted(
            ((key, float(value)) for key, value in salience.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_salience = ", ".join(f"{key}={value:.2f}" for key, value in sorted_salience[:5])
        self.scratchpad.append(f"step {self.step_index} reflection: {top_salience}")
        patterns = self.pattern_library.retrieve(salience)
        if not patterns and self.pattern_library.patterns:
            patterns = self.pattern_library.patterns[:1]
        for pattern in patterns:
            self.scratchpad.append(f"pattern::{pattern.name} -> {pattern.description}")
        meta_snapshot = self.meta_state.snapshot()
        self.scratchpad.append(
            "meta state: confidence={:.2f} roi={:.2f}".format(
                float(meta_snapshot.get("confidence", 0.0)),
                float(meta_snapshot.get("roi", 0.0)),
            )
        )
        if self.workspace_viewer:
            listing = self.introspection.get_workspace_listing(".")
            if listing:
                focus_paths = ", ".join(item["path"] for item in listing[:3])
                if focus_paths:
                    self.scratchpad.append(f"workspace focus: {focus_paths}")
        context_snippet = state.get("context_snippet")
        if isinstance(context_snippet, str) and context_snippet.strip():
            self.scratchpad.append(f"context noted: {context_snippet.strip()[:160]}")
        metadata = {
            "step": self.step_index,
            "salience": {key: value for key, value in sorted_salience[:6]},
            "patterns": [pattern.name for pattern in patterns],
        }
        self.scratchpad.commit(outcome=True, metadata=metadata)
        for pattern in patterns:
            self.pattern_library.log_usage(
                pattern,
                success=True,
                benefit=float(salience.get("progress", 0.0)),
                cost=float(state.get("token_cost", 1.0)),
            )
        reward = (
            0.3 * float(salience.get("novelty", 0.0))
            + 0.2 * float(salience.get("uncertainty", 0.0))
            + 0.1 * float(salience.get("progress", 0.0))
            - 0.15 * float(salience.get("drag", 0.0))
        )
        self.bandit_trainer.update(action, reward)

    def _enrich_state_for_sensors(self, state: Mapping[str, object]) -> Dict[str, object]:
        enriched: Dict[str, object] = dict(state)
        last_action = self.controller.state.last_action
        if last_action is not None:
            enriched.setdefault("last_action", last_action.operator.name)
            enriched.setdefault("last_action_depth", last_action.cot_depth)
        if "scratchpad_text" not in enriched:
            if self.scratchpad.current_trace:
                enriched["scratchpad_text"] = " | ".join(self.scratchpad.current_trace)
            else:
                enriched["scratchpad_text"] = self.scratchpad.summarize(max_traces=1)
        if "scratchpad" not in enriched:
            enriched["scratchpad"] = list(self.scratchpad.current_trace)
        if "contradictions" not in enriched:
            enriched["contradictions"] = 0.0
        if self.hidden_states is not None and "embedding" not in enriched:
            enriched["embedding"] = self._hidden_embedding_vector(self.hidden_states)
        return enriched

    @staticmethod
    def _hidden_embedding_vector(tensor: torch.Tensor) -> object:
        try:
            return tensor.detach().cpu().numpy().flatten()[-256:]
        except Exception:
            return []
