"""Runtime orchestrator wiring the salience subsystems together."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Mapping, Optional, Sequence

import numpy as np

from ..core.controller import (
    BanditTrainer,
    ControllerAction,
    ControllerDecision,
    ControllerOperator,
    ControllerPatch,
    SalienceControllerPolicy,
)
from ..core.meta import EpisodicStore, MetaState, build_episode, render_self_report
from ..core.memory import StructuredMemory
from ..core.operators import (
    AuctionBid,
    ComputeAuction,
    GraphReasoner,
    MemoryOperator,
    SparseJumpTeleporter,
    VerifierSuite,
    SASSCore,
)
from ..core.ideas import ExperimentDispatcher, IdeaDispatcher, IdeaGenerator, IdeaSimulator
from ..core.scheduler import EventDrivenScheduler
from ..core.reflection import (
    IntrospectionInterface,
    PatternLibrary,
    Scratchpad,
    WorkspaceViewer,
)
from ..telemetry import BUS, TelemetryEvent

from .action_executor import (
    ActionContext,
    ActionExecutor,
    MCPToolSession,
    MemoryActionHandler,
    ReflectionActionHandler,
    SassActionHandler,
    ToolActionHandler,
    ToolInvocationAdapter,
    VerifyActionHandler,
)
from .config import RuntimeConfig
from .experiments import ExperimentCoordinator
from .maintenance import MaintenanceManager, default_archive_store
from .mcp_bridge import IntrospectionResource, MemoryResource
from .reflection import ReflectionManager
from .sensor_pipeline import SensorPipeline


@dataclass(slots=True)
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


class SalienceRuntime:
    """Main runtime driver for SalienceOS Seed."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.memory = StructuredMemory()
        self.sensor_pipeline = SensorPipeline()
        salience_dim = self.sensor_pipeline.salience_dim
        if self.config.meta.salience_dim != salience_dim:
            self.config.meta.salience_dim = salience_dim
        self.meta_state = MetaState(self.config.meta)
        self.meta_state.vector[self.config.meta.confidence_index] = 0.5
        self.meta_state.vector[self.config.meta.roi_index] = 0.3

        self.sass = SASSCore(self.config.sass)
        self.teleporter = SparseJumpTeleporter()
        self.graph_reasoner = GraphReasoner(self.config.graph_reasoner)
        self.verifier = VerifierSuite()

        bandit_store: Dict[str, Dict[str, float]] = {}
        self.bandit_trainer = BanditTrainer(self.config.controller_bandit, bandit_store)
        self.controller = SalienceControllerPolicy(self.config.controller, bandit_store)

        self.scheduler = EventDrivenScheduler(config=self.config.scheduler)
        self.auction = ComputeAuction()

        self.idea_generator = IdeaGenerator(self.config.idea_factory)
        self.idea_simulator = IdeaSimulator()
        self.idea_dispatcher = IdeaDispatcher(self.memory)

        self.memory_operator = MemoryOperator(self.memory)

        self.workspace_viewer = (
            WorkspaceViewer(self.config.reflection.workspace_root)
            if self.config.reflection.workspace_root
            else None
        )
        self.scratchpad = Scratchpad(
            max_tokens=self.config.reflection.scratchpad_tokens,
            history_capacity=self.config.reflection.history_capacity,
        )
        self.pattern_library = PatternLibrary()
        self.introspection = IntrospectionInterface(
            controller=self.controller,
            verifier=self.verifier,
            memory=self.memory,
            meta_state=self.meta_state,
            workspace=self.workspace_viewer,
        )
        self.reflection_manager = ReflectionManager(
            scratchpad=self.scratchpad,
            pattern_library=self.pattern_library,
            introspection=self.introspection,
            workspace_viewer=self.workspace_viewer,
            meta_state=self.meta_state,
        )

        self.action_context = ActionContext()
        self.runtime_tools: Dict[str, object] = {}
        self.tool_adapter = ToolInvocationAdapter(
            runtime_tools=self.runtime_tools,
            operator_tool_map=self._default_tool_map(),
        )
        lambda_cost = self.config.controller.lambda_cost
        self.action_executor = ActionExecutor(
            verify_handler=VerifyActionHandler(
                verifier=self.verifier,
                bandit=self.bandit_trainer,
                introspection=self.introspection,
                memory=self.memory,
                lambda_cost=lambda_cost,
            ),
            sass_handler=SassActionHandler(
                sass=self.sass,
                teleporter=self.teleporter,
                graph_reasoner=self.graph_reasoner,
                memory=self.memory,
                bandit=self.bandit_trainer,
                context=self.action_context,
                lambda_cost=lambda_cost,
            ),
            memory_handler=MemoryActionHandler(
                memory_operator=self.memory_operator,
                bandit=self.bandit_trainer,
            ),
            tool_handler=ToolActionHandler(
                bandit=self.bandit_trainer,
                tool_adapter=self.tool_adapter,
            ),
            reflection_handler=ReflectionActionHandler(
                reflection=self.reflection_manager,
                bandit=self.bandit_trainer,
            ),
        )

        self.maintenance_manager = None
        if self.config.maintenance.enabled:
            archive_store = default_archive_store()
            self.maintenance_manager = MaintenanceManager(
                memory=self.memory,
                thresholds=self.config.maintenance.thresholds,
                store=archive_store,
            )

        self.experiment_dispatcher = (
            ExperimentDispatcher(
                self.config.experiments.parameters,
                max_concurrent=self.config.experiments.max_concurrent,
            )
            if self.config.experiments.enabled
            else None
        )
        self.experiments = ExperimentCoordinator(
            dispatcher=self.experiment_dispatcher,
            duration=self.config.experiments.duration,
        )

        self.budget_left = float(self.config.budget_tokens)
        self.step_index = 0
        self._verification_history: list[int] = []
        self._verification_rate = 0.0
        self._dynamics_history: list[Dict[str, object]] = []
        self._salience_history: Dict[str, Deque[float]] = {}
        self._salience_history_window = 128
        self._salience_histogram_bins = 12
        self._last_auction_results: Dict[str, float] = {}
        self.sensor_bank = self.sensor_pipeline.sensor_bank

        episodic_path = None
        if self.config.episodic.enabled:
            if self.config.episodic.store_path:
                episodic_path = Path(self.config.episodic.store_path).expanduser()
        self.episodic_store = EpisodicStore(episodic_path) if self.config.episodic.enabled else None

        self.mcp_memory = MemoryResource(self.memory_operator)
        self.mcp_introspection = IntrospectionResource(self.introspection)

    def update_sensor_context(self, key: str, payload: Mapping[str, object]) -> None:
        """Expose additional context to the sensor pipeline in fallback mode."""

        self.sensor_pipeline.update_memory_snapshot({key: payload})

    def _default_tool_map(self) -> Mapping[ControllerOperator, Mapping[ControllerPatch, str]]:
        return {
            ControllerOperator.TOOL: {
                ControllerPatch.MATH: "math",
                ControllerPatch.RETRIEVAL: "retrieval",
                ControllerPatch.PLAN: "plan",
            }
        }

    def set_training_active(self, active: bool) -> None:
        """Toggle whether the runtime should preserve autograd history between steps."""

        self.action_context.training_active = bool(active)

    def _execute_action(
        self,
        decision: ControllerDecision,
        state: Mapping[str, object],
        salience: Mapping[str, float],
    ) -> Optional[bool]:  # pragma: no cover - exercised via tests
        return self.action_executor.execute(decision, state, salience)

    @property
    def hidden_states(self):  # pragma: no cover - exercised via tests
        return self.action_context.hidden_states

    def attach_mcp_tool_client(self, client) -> MCPToolSession:
        session = MCPToolSession(client)
        self.tool_adapter.register_mcp_session(session)
        return session

    def mcp_resources(self) -> Mapping[str, object]:
        return {
            "memory": self.mcp_memory,
            "introspection": self.mcp_introspection,
        }

    def run_step(self, state: Mapping[str, object]) -> RuntimeMetrics:
        """Execute a single runtime iteration."""

        enriched_state = self.sensor_pipeline.enrich_state(
            state,
            scratchpad=self.scratchpad,
            hidden_states=self.action_context.hidden_states,
            controller_last_action=self.controller.state.last_action,
        )
        meta_snapshot = dict(self.meta_state.snapshot())
        meta_snapshot.setdefault("verification_pass_rate", self._verification_rate)
        structured_snapshot = self.memory.as_runtime_mapping()
        self.introspection.attach_memory_snapshot(structured_snapshot)
        self.sensor_pipeline.update_memory_snapshot(structured_snapshot)

        salience_map, salience_vector = self.sensor_pipeline.run(enriched_state, meta_snapshot)
        self.introspection.attach_salience(salience_map)

        self._prepare_auction_bids(salience_map, meta_snapshot)
        previous_cooldown = self.controller.state.cooldown_remaining
        decision = self._choose_with_defaults(salience_map, meta_snapshot)
        should_run = self.scheduler.should_fire(
            salience_map,
            decision_operator=decision.action.operator.name,
            budget_left=self.budget_left,
            budget_total=self.config.budget_tokens,
        )
        verification_passed: Optional[bool] = None
        if should_run:
            enriched_state["step_index"] = self.step_index
            verification_passed = self.action_executor.execute(decision, enriched_state, salience_map)

        accepted_ideas = self._maybe_generate_ideas(salience_map, meta_snapshot)

        self.step_index += 1
        self.budget_left = max(0.0, self.budget_left - state.get("token_cost", 1.0))
        self.meta_state.update(
            salience_vector=salience_map,
            verification_passed=verification_passed,
            budget_left=self.budget_left,
            cooldown_active=self.controller.state.cooldown_remaining > 0,
        )

        self._update_verification_history(verification_passed)
        cleanup_report = self.maintenance_manager.run(salience_map) if self.maintenance_manager else {}
        self.experiments.propose(salience_map, verification_passed, self._apply_overrides)
        experiment_reports = self.experiments.advance(
            salience_map,
            verification_passed,
            overrides_reverter=self._revert_overrides,
        )
        episode_id = self._record_episode(salience_map, decision, verification_passed)
        yearning_snapshot = self.introspection.get_yearning_state(refresh=True)
        meta_after = self.meta_state.snapshot()

        metrics = RuntimeMetrics(
            step=self.step_index,
            decision=decision,
            meta_report=render_self_report(meta_after),
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
        self._record_salience_history(salience_vector)
        self._publish_salience_histogram(metrics.step, salience_vector)
        self._publish_controller_trace(
            metrics,
            meta_before=meta_snapshot,
            meta_after=meta_after,
            previous_cooldown=previous_cooldown,
            executed=should_run,
        )
        return metrics

    def adjust_controller_dynamics(self, updates: Mapping[str, float], *, max_step: float = 0.1) -> Dict[str, float]:
        applied = self.introspection.adjust_controller_dynamics(updates, max_step=max_step)
        if applied:
            self._dynamics_history.append(
                {
                    "step": self.step_index,
                    "applied": dict(applied),
                }
            )
            if len(self._dynamics_history) > 64:
                self._dynamics_history.pop(0)
        return applied

    def dynamics_history(self) -> Sequence[Dict[str, object]]:
        return list(self._dynamics_history)

    def _maybe_generate_ideas(self, salience: Mapping[str, float], meta_snapshot: Mapping[str, float]) -> int:
        if not self.idea_generator.should_generate(salience):
            return 0
        proposals = self.idea_generator.generate(salience, meta_snapshot, self.memory)
        if not proposals:
            return 0
        simulations = self.idea_simulator.simulate(proposals, meta_snapshot, salience)
        accepted = self.idea_dispatcher.dispatch(simulations, meta_snapshot)
        return len(accepted)

    def _record_salience_history(self, salience_vector) -> None:
        for reading in salience_vector.readings:
            history = self._salience_history.get(reading.name)
            if history is None or history.maxlen != self._salience_history_window:
                history = deque(maxlen=self._salience_history_window)
                self._salience_history[reading.name] = history
            value = getattr(reading, "normalised", getattr(reading, "raw", 0.0))
            history.append(float(value))

    def _publish_salience_histogram(self, step: int, salience_vector) -> None:
        hist_payload: Dict[str, Dict[str, object]] = {}
        for reading in salience_vector.readings:
            history = self._salience_history.get(reading.name)
            if not history:
                continue
            snapshot = self._build_histogram_snapshot(history)
            if not snapshot:
                continue
            entry: Dict[str, object] = dict(snapshot)
            latest = getattr(reading, "normalised", getattr(reading, "raw", 0.0))
            raw_value = getattr(reading, "raw", latest)
            entry["latest"] = float(latest)
            entry["raw"] = float(raw_value)
            metadata = getattr(reading, "metadata", None)
            if metadata:
                entry["metadata"] = self._serialise_mapping(metadata)
            hist_payload[reading.name] = entry
        if not hist_payload:
            return
        BUS.publish(
            TelemetryEvent(
                type="runtime/salience_histogram",
                payload={
                    "step": int(step),
                    "histograms": hist_payload,
                },
            )
        )

    def _build_histogram_snapshot(self, history: Deque[float]) -> Dict[str, object]:
        if not history:
            return {}
        arr = np.fromiter(history, dtype=np.float32)
        if arr.size == 0:
            return {}
        counts, edges = np.histogram(arr, bins=self._salience_histogram_bins)
        bins = [
            {
                "lower": float(edges[idx]),
                "upper": float(edges[idx + 1]),
                "count": int(count),
            }
            for idx, count in enumerate(counts)
        ]
        return {
            "samples": int(arr.size),
            "mean": float(np.mean(arr)),
            "stdev": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "bins": bins,
        }

    def _publish_controller_trace(
        self,
        metrics: RuntimeMetrics,
        *,
        meta_before: Mapping[str, float],
        meta_after: Mapping[str, float],
        previous_cooldown: int,
        executed: bool,
    ) -> None:
        score_components = getattr(self.controller, "_last_score_components", {})
        score_trace = {
            self.controller._action_key(action): {
                "score": float(score),
                "rationale": rationale,
            }
            for action, (score, rationale) in score_components.items()
        }
        bids_sorted = sorted(self._last_auction_results.items(), key=lambda item: item[1], reverse=True)
        payload = {
            "step": int(metrics.step),
            "decision": {
                "action": self._action_payload(metrics.decision.action),
                "score": float(metrics.decision.score),
                "hysteresis_delta": float(metrics.decision.hysteresis_delta),
                "cooldown": {
                    "previous": int(previous_cooldown),
                    "current": int(metrics.decision.cooldown_steps),
                },
            },
            "executed": bool(executed),
            "verification_passed": metrics.verification_passed,
            "budget_left": float(metrics.budget_left),
            "salience": {name: float(value) for name, value in metrics.decision.salience_mapping.items()},
            "score_trace": score_trace,
            "auction": [
                {"action": action_key, "score": float(score)}
                for action_key, score in bids_sorted[:8]
            ],
            "meta": {
                "before": self._serialise_mapping(meta_before),
                "after": self._serialise_mapping(meta_after),
            },
        }
        BUS.publish(
            TelemetryEvent(
                type="runtime/controller_trace",
                payload=payload,
            )
        )

    @staticmethod
    def _action_payload(action: ControllerAction) -> Dict[str, object]:
        return {
            "operator": action.operator.name,
            "cot_depth": int(action.cot_depth),
            "patch": action.patch.name,
        }

    @staticmethod
    def _serialise_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
        serialised: Dict[str, object] = {}
        for key, value in mapping.items():
            serialised[key] = SalienceRuntime._serialise_value(value)
        return serialised

    @staticmethod
    def _serialise_value(value: object) -> object:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (float, int)):
            return value
        return value

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
        resolved = self.auction.resolve()
        self._last_auction_results = {
            self.controller._action_key(action): float(score)
            for action, score in resolved.items()
        }
        for action, score in resolved.items():
            self.controller.register_auction_bid(action, score)
        self.auction.clear()

    def _choose_with_defaults(self, salience_map: Mapping[str, float], meta_snapshot: Mapping[str, float]) -> ControllerDecision:
        decision = self.controller.choose(salience_map, meta_snapshot, self.budget_left)
        action = decision.action
        cost_pressure = 1.0 - min(0.99, self.budget_left / max(self.config.budget_tokens, 1.0))
        fatigue = float(meta_snapshot.get("fatigue", 0.0))
        allow_train = self._should_allow_training(salience_map, meta_snapshot, cost_pressure, fatigue)
        if action.operator is ControllerOperator.REFLECT:
            return decision
        if action.operator is ControllerOperator.SASS and action.cot_depth < 2:
            patched_action = ControllerAction(cot_depth=2, operator=ControllerOperator.REFLECT, patch=action.patch)
            patched_decision = ControllerDecision(
                action=patched_action,
                score=decision.score,
                salience_mapping=decision.salience_mapping,
                cooldown_steps=decision.cooldown_steps,
                hysteresis_delta=decision.hysteresis_delta,
            )
            self._sync_controller_state_with_decision(patched_decision)
            return patched_decision
        if action.operator not in {ControllerOperator.REFLECT, ControllerOperator.VERIFY} and not allow_train:
            patched_action = ControllerAction(
                cot_depth=max(action.cot_depth, 2),
                operator=ControllerOperator.REFLECT,
                patch=ControllerPatch.NONE,
            )
            patched_decision = ControllerDecision(
                action=patched_action,
                score=decision.score,
                salience_mapping=decision.salience_mapping,
                cooldown_steps=decision.cooldown_steps,
                hysteresis_delta=decision.hysteresis_delta,
            )
            self._sync_controller_state_with_decision(patched_decision)
            return patched_decision
        return decision

    def _sync_controller_state_with_decision(self, decision: ControllerDecision) -> None:
        """Align controller state to a patched decision."""

        state = self.controller.state
        state.last_action = decision.action
        state.last_score = decision.score
        state.cooldown_remaining = decision.cooldown_steps

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

    def _update_verification_history(self, outcome: Optional[bool]) -> None:
        if outcome is None:
            return
        self._verification_history.append(1 if outcome else 0)
        if len(self._verification_history) > 64:
            self._verification_history.pop(0)
        if self._verification_history:
            self._verification_rate = sum(self._verification_history) / len(self._verification_history)

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
            "verification_passed"
            if verification_passed
            else "verification_failed"
            if verification_passed is False
            else "no_verification"
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

    def _apply_overrides(self, overrides: Mapping[str, float]) -> Dict[str, float]:
        originals: Dict[str, float] = {}
        for key, value in overrides.items():
            target, attr = self._resolve_override_target(key)
            originals[key] = float(getattr(target, attr))
            setattr(target, attr, float(value))
        return originals

    def _revert_overrides(self, originals: Mapping[str, float]) -> None:
        for key, value in originals.items():
            target, attr = self._resolve_override_target(key)
            setattr(target, attr, float(value))

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


__all__ = ["RuntimeMetrics", "SalienceRuntime"]
