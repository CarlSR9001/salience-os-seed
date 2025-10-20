"""Adaptive coordinator bridging calibration probes with controller heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ..core.controller.policy import ControllerConfig
from ..core.controller.s_prime import SPrimeConfig
from ..proto_lm.trainer import ProtoLanguageModel
from ..runtime.orchestrator import RuntimeMetrics, SalienceRuntime
from ..runtime.sensors.calibration import SensorCalibrationSuite

from .gradient_flow import AdaptiveGradientFlow, FlowSignal
from .vault import AdaptiveVault, WeightProvenance
from .weight_learner import AdaptiveWeightLearner, SalienceWeights
from .axioms import AxiomGuard, AxiomViolation
from .elegance import EleganceCandidate, EleganceConfig, EleganceJudge, EleganceMetrics
from .truth import TruthDecision, TruthGuard


@dataclass
class GatingSummary:
    truth_decision: TruthDecision
    truth_star: float
    combined_score: float
    axiom_violations: tuple[AxiomViolation, ...]
    elegance_accept: Optional[bool]
    elegance_score: Optional[float]


class AdaptiveCoordinator:
    """Bridges model training signals with salience controllers and guards."""

    def __init__(
        self,
        *,
        runtime: SalienceRuntime,
        proto_lm: ProtoLanguageModel,
    ) -> None:
        self.runtime = runtime
        self.proto_lm = proto_lm
        controller_cfg: ControllerConfig = runtime.controller.config
        s_prime_cfg: SPrimeConfig = runtime.controller.s_prime.config
        base_weights = SalienceWeights(
            novelty=max(0.05, min(0.9, controller_cfg.delta_weight)),
            retention=max(0.05, min(0.9, controller_cfg.key_weight)),
            payoff=max(0.05, min(0.9, controller_cfg.aim_weight)),
        ).normalized()

        self._baseline_weights = base_weights
        calibration_cfg = getattr(runtime.config, "calibration", None)
        self._calibration_suite: SensorCalibrationSuite | None = None
        self._calibration_channels: Mapping[str, str] = {
            "novelty": "novelty",
            "retention": "progress",
            "payoff": "roi",
        }
        self._probe_weight = float(getattr(calibration_cfg, "probe_weight", 0.0)) if calibration_cfg else 0.0
        self._heuristic_regularization = (
            float(getattr(calibration_cfg, "heuristic_regularization", 0.0))
            if calibration_cfg
            else 0.0
        )
        if calibration_cfg and getattr(calibration_cfg, "enabled", False):
            self._calibration_suite = SensorCalibrationSuite(
                baselines={
                    "novelty": base_weights.novelty,
                    "retention": base_weights.retention,
                    "payoff": base_weights.payoff,
                },
                history_window=getattr(calibration_cfg, "history_window", 128),
                ridge_penalty=getattr(calibration_cfg, "ridge_penalty", 1e-2),
                min_samples=getattr(calibration_cfg, "min_samples", 8),
            )

        self.vault = AdaptiveVault()
        self.gradient_flow = AdaptiveGradientFlow()
        self.weight_learner = AdaptiveWeightLearner({"controller": base_weights})
        self.truth_guard = TruthGuard()
        self.elegance_judge = EleganceJudge(EleganceConfig())
        self.axiom_guard = AxiomGuard()
        self._controller_cfg = controller_cfg
        self._sprime_cfg = s_prime_cfg
        self._last_truth_decision: Optional[str] = None
        self._last_gating_summary: Optional[GatingSummary] = None
        self.proto_lm.add_training_observer(self.observe_training)

    @property
    def last_gating_summary(self) -> Optional[GatingSummary]:
        return self._last_gating_summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, object]:
        summary_payload: Optional[Dict[str, object]] = None
        if self._last_gating_summary is not None:
            summary_payload = {
                "truth_decision": self._last_gating_summary.truth_decision.decision,
                "action_score": self._last_gating_summary.truth_decision.action_score,
                "combined_score": self._last_gating_summary.truth_decision.combined_score,
                "reason": self._last_gating_summary.truth_decision.reason,
                "truth_star": self._last_gating_summary.truth_star,
                "combined": self._last_gating_summary.combined_score,
                "axioms": [violation.axiom_id for violation in self._last_gating_summary.axiom_violations],
                "elegance_accept": self._last_gating_summary.elegance_accept,
                "elegance_score": self._last_gating_summary.elegance_score,
            }
        state: Dict[str, object] = {
            "vault": self.vault.serialize(),
            "gradient_flow": self.gradient_flow.state_dict(),
            "weight_learner": self.weight_learner.state_dict(),
            "last_truth_decision": self._last_truth_decision,
            "gating_summary": summary_payload,
        }
        return state

    def import_state(self, payload: Dict[str, object]) -> None:
        vault_payload = payload.get("vault")
        if isinstance(vault_payload, dict):
            self.vault.restore(vault_payload)
        gradient_payload = payload.get("gradient_flow")
        if isinstance(gradient_payload, dict):
            self.gradient_flow.load_state_dict(gradient_payload)
        weight_payload = payload.get("weight_learner")
        if isinstance(weight_payload, dict):
            self.weight_learner.load_state_dict(weight_payload)
            controller_weights = self.weight_learner.weights().get("controller")
            if controller_weights is not None:
                self._apply_weights(controller_weights)
        self._last_truth_decision = payload.get("last_truth_decision") if isinstance(payload.get("last_truth_decision"), str) else None
        summary_payload = payload.get("gating_summary")
        if isinstance(summary_payload, dict):
            self._last_gating_summary = self._deserialize_gating_summary(summary_payload)

    def _deserialize_gating_summary(self, payload: Dict[str, object]) -> Optional[GatingSummary]:
        decision_name = payload.get("truth_decision")
        if not isinstance(decision_name, str):
            return None
        decision = TruthDecision(
            action_score=float(payload.get("action_score", 0.0)),
            combined_score=float(payload.get("combined_score", 0.0)),
            decision=decision_name,
            reason=str(payload.get("reason", "")),
        )
        truth_star = float(payload.get("truth_star", 0.0))
        combined = float(payload.get("combined", 0.0))
        axiom_ids = payload.get("axioms", [])
        violations: list[AxiomViolation] = []
        if isinstance(axiom_ids, list):
            for axiom_id in axiom_ids:
                if isinstance(axiom_id, str):
                    violations.append(
                        AxiomViolation(axiom_id=axiom_id, reason="restored", context={})
                    )
        elegance_accept = payload.get("elegance_accept")
        if elegance_accept is not None and not isinstance(elegance_accept, bool):
            elegance_accept = None
        elegance_score = payload.get("elegance_score")
        if elegance_score is not None:
            elegance_score = float(elegance_score)
        return GatingSummary(
            truth_decision=decision,
            truth_star=truth_star,
            combined_score=combined,
            axiom_violations=tuple(violations),
            elegance_accept=elegance_accept if isinstance(elegance_accept, (bool, type(None))) else None,
            elegance_score=elegance_score,
        )

    # ------------------------------------------------------------------
    # Training integration
    # ------------------------------------------------------------------
    def observe_training(self, snapshot: Dict[str, object]) -> None:
        loss = float(snapshot.get("loss", 0.0))
        loss_components = snapshot.get("loss_components", {})
        loss_old = float(loss_components.get("old", loss))
        loss_new = float(loss_components.get("new", loss))
        grad_health = snapshot.get("grad_health", {})
        novelty = 0.0 if loss <= 0.0 else min(1.0, max(0.0, loss_new / max(loss, 1e-6)))
        retention = min(1.0, max(0.0, 1.0 - loss_new / max(loss_old + 1e-6, 1.0)))
        payoff = min(1.0, max(0.0, 1.0 - loss))
        salience_scores = {
            "novelty": novelty,
            "retention": retention,
            "payoff": payoff,
            "cost": float(grad_health.get("grad_norm", 0.0)) / 10.0,
        }
        notes = {
            "loss": loss,
            "loss_old": loss_old,
            "loss_new": loss_new,
            "grad_health": grad_health,
        }
        self.vault.register(
            payload={"training_snapshot": snapshot},
            salience_scores=salience_scores,
            provenance=WeightProvenance.AUTONOMOUS,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Runtime integration
    # ------------------------------------------------------------------
    def track_runtime(self, metrics: RuntimeMetrics) -> None:
        decision = metrics.decision
        salience = decision.salience_mapping
        reward = float(decision.score)
        penalty = float(salience.get("drag", 0.0) + salience.get("cost", 0.0))
        components = {
            "novelty": float(salience.get("novelty", 0.0)),
            "retention": float(salience.get("progress", 0.0)),
            "payoff": float(salience.get("roi", 0.0)),
            "cost": float(salience.get("cost", 0.0)),
        }
        raw_inputs = self._record_calibration(metrics, components)
        metadata = {
            "step": metrics.step,
            "verification": 1.0 if metrics.verification_passed else 0.0,
            "operator": decision.action.operator.name,
        }
        signal = FlowSignal(
            task="controller",
            reward=reward,
            penalty=penalty,
            components=components,
            metadata=metadata,
        )
        self.gradient_flow.record(signal)
        if not self.gradient_flow.has_signals("controller"):
            return
        estimate = self.gradient_flow.drain("controller")
        if not estimate:
            return
        avg_reward = estimate.avg_reward
        metrics_payload = {
            "quality": max(0.0, min(1.0, 0.5 + 0.1 * avg_reward)),
            "satisfaction": max(0.0, min(1.0, 0.5 + 0.05 * estimate.net)),
        }
        calibration_weights, calibration_confidence = self._predict_calibration(raw_inputs)
        new_weights = self.weight_learner.consider_update(
            "controller",
            metrics_payload,
            gradients=estimate.gradients,
        )
        if new_weights and calibration_weights is not None:
            new_weights = self._blend_with_calibration(new_weights, calibration_weights, calibration_confidence)
        if new_weights:
            self._apply_weights(new_weights)

    def _record_calibration(
        self,
        metrics: RuntimeMetrics,
        components: Mapping[str, float],
    ) -> Dict[str, float]:
        if not self._calibration_suite:
            return {}
        raw_inputs: Dict[str, float] = {}
        raw_salience = dict(getattr(metrics, "salience_raw", {}) or {})
        reward = float(metrics.decision.score)
        success = 1.0 if metrics.verification_passed else 0.0
        for channel, sensor_key in self._calibration_channels.items():
            raw_value = raw_salience.get(sensor_key)
            if raw_value is None:
                raw_value = metrics.decision.salience_mapping.get(sensor_key, 0.0)
            raw_float = float(raw_value)
            target = float(components.get(channel, 0.0))
            self._calibration_suite.observe(
                channel,
                raw_value=raw_float,
                target_weight=max(0.0, min(1.0, target)),
                reward=reward,
                success=success,
            )
            raw_inputs[channel] = raw_float
        return raw_inputs

    def _predict_calibration(self, raw_inputs: Mapping[str, float]) -> tuple[Optional[SalienceWeights], float]:
        if not self._calibration_suite or not raw_inputs:
            return None, 0.0
        weights, confidences = self._calibration_suite.predict(raw_inputs)
        if not weights:
            return None, 0.0
        novelty = float(weights.get("novelty", self._baseline_weights.novelty))
        retention = float(weights.get("retention", self._baseline_weights.retention))
        payoff = float(weights.get("payoff", self._baseline_weights.payoff))
        calibrated = SalienceWeights(novelty=novelty, retention=retention, payoff=payoff).normalized()
        confidence_values = list(confidences.values())
        confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
        return calibrated, confidence

    def _blend_with_calibration(
        self,
        proposal: SalienceWeights,
        calibrated: SalienceWeights,
        confidence: float,
    ) -> SalienceWeights:
        if confidence <= 0.0:
            return proposal
        probe_mix = max(0.0, min(1.0, self._probe_weight * confidence))
        blended = SalienceWeights(
            novelty=(1.0 - probe_mix) * proposal.novelty + probe_mix * calibrated.novelty,
            retention=(1.0 - probe_mix) * proposal.retention + probe_mix * calibrated.retention,
            payoff=(1.0 - probe_mix) * proposal.payoff + probe_mix * calibrated.payoff,
        ).normalized()
        if self._heuristic_regularization <= 0.0:
            return blended
        reg = max(0.0, min(1.0, self._heuristic_regularization))
        regularized = SalienceWeights(
            novelty=reg * self._baseline_weights.novelty + (1.0 - reg) * blended.novelty,
            retention=reg * self._baseline_weights.retention + (1.0 - reg) * blended.retention,
            payoff=reg * self._baseline_weights.payoff + (1.0 - reg) * blended.payoff,
        ).normalized()
        return regularized

    def assess_response(self, response: str, metrics: RuntimeMetrics) -> tuple[str, GatingSummary]:
        decision = metrics.decision
        salience = decision.salience_mapping
        salience_raw = getattr(metrics, "salience_raw", {}) or {}
        truth = float(salience_raw.get("truth", salience.get("truth", salience.get("alignment", 0.0))))
        uncertainty = float(salience_raw.get("uncertainty", salience.get("uncertainty", salience.get("entropy", 0.0))))
        contradiction = float(salience_raw.get("contradictions", salience.get("contradictions", 0.0)))
        risk = float(salience_raw.get("risk", salience.get("risk", salience.get("cost", 0.0))))
        truth_star = self.truth_guard.truth_score(truth, uncertainty, contradiction, risk)
        combined = self.truth_guard.fuse(decision.score, truth_star)
        truth_decision = self.truth_guard.decide(
            action_score=decision.score,
            combined_score=combined,
            truth_star=truth_star,
            previous=self._last_truth_decision,
        )
        self._last_truth_decision = truth_decision.decision

        axiom_violations = tuple(self.axiom_guard.evaluate_speak(truth_star=truth_star, combined_score=combined))

        elegance_result = None
        elegance_score = None
        best_snapshot = self.vault.best_candidate()
        if best_snapshot is not None:
            notes = best_snapshot.notes
            loss_prev = float(notes.get("loss_old", 0.0))
            loss_now = float(notes.get("loss_new", 0.0))
            delta_mdl = loss_prev - loss_now
            regressions = max(0.0, loss_now - loss_prev)
            elegance_metrics = EleganceMetrics(
                delta_mdl=delta_mdl,
                regressions=regressions,
                transfer_gain=float(salience.get("progress", 0.0)),
                orthogonal_novelty=float(salience.get("novelty", 0.0)),
                tstar_min=truth_star,
                risk=risk,
                validators_passed=1 if metrics.verification_passed else 0,
            )
            candidate = EleganceCandidate(
                weight_id=best_snapshot.weight_id,
                description="training_snapshot",
                baseline_bits=loss_prev,
                operations=("online_update",),
            )
            elegance_result = self.elegance_judge.evaluate(candidate, elegance_metrics)
            elegance_score = elegance_result.score

        summary = GatingSummary(
            truth_decision=truth_decision,
            truth_star=truth_star,
            combined_score=combined,
            axiom_violations=axiom_violations,
            elegance_accept=None if elegance_result is None else elegance_result.accepted,
            elegance_score=elegance_score,
        )
        self._last_gating_summary = summary

        if truth_decision.decision == "DROP" and not axiom_violations:
            response = (
                "I'm pausing to double-check the facts before continuing. "
                "(truth_star={:.2f}, combined_score={:.2f})".format(truth_star, combined)
            )
        elif axiom_violations:
            violated = ", ".join(violation.axiom_id for violation in axiom_violations)
            response = (
                "I need to revisit that answer because it conflicts with internal safety axioms: {}."
                .format(violated)
            )
        return response, summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_weights(self, weights: SalienceWeights) -> None:
        normalized = weights.normalized()
        self._controller_cfg.delta_weight = max(0.1, normalized.novelty * 2.0)
        self._controller_cfg.key_weight = max(0.1, normalized.retention * 2.0)
        self._controller_cfg.aim_weight = max(0.1, normalized.payoff * 2.0)
        self._sprime_cfg.w_novelty = normalized.novelty
        self._sprime_cfg.w_retention = normalized.retention
        self._sprime_cfg.w_payoff = normalized.payoff
