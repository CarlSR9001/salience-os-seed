"""S′ controller scoring built on salience-native metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import math

from .actions import ControllerAction, ControllerOperator


@dataclass
class SPrimeConfig:
    """Hyperparameters governing the S′ scoring function."""

    w_novelty: float = 0.30
    w_retention: float = 0.35
    w_payoff: float = 0.35
    novelty_alpha: float = 1.0
    novelty_beta: float = 0.1
    discount_rate: float = 0.1
    fatigue_k: float = 0.2
    fatigue_gamma: float = 1.0
    cost_lambda: float = 0.015
    meta_confidence_boost: float = 0.10
    meta_roi_boost: float = 0.05
    depth_factor_scale: float = 0.08
    yearning_enabled: bool = True
    yearning_rho: float = 0.12
    yearning_sigma: float = 0.6
    yearning_alpha: float = 0.18
    yearning_beta: float = 0.32
    yearning_epsilon: float = 0.05
    dynamics: "DynamicsConfig" = field(default_factory=lambda: DynamicsConfig())


@dataclass
class DynamicsConfig:
    lambda_affinity: float = 0.12
    xi_affinity: float = 0.35
    pi_scarcity: float = 0.18
    pi_novelty: float = 0.25
    pi_spend: float = 0.3
    accelerant_intensity: float = 0.4
    mu_exposure: float = 0.25
    nu_external: float = 0.05
    lambda_fatigue: float = 0.85
    zeta_volume: float = 0.12
    alpha_quality: float = 1.0
    beta_scarcity: float = 1.0
    beta_coherence: float = 1.0
    gamma_memory: float = 1.0
    rho_novelty: float = 0.6
    kappa_base: float = 0.3
    kappa_trend: float = 0.25
    psi_cost: float = 0.15
    delta_distance: float = 1.0
    delta_safe: float = 1.15
    chi_premature: float = 0.4
    omega_exposure: float = 0.35
    eta_fatigue: float = 0.3
    trend_key: str = "trend"
    distance_key: str = "distance_to_payoff"
    epsilon: float = 1e-3


DYNAMICS_FIELD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lambda_affinity": (0.0, 1.0),
    "xi_affinity": (0.0, 2.0),
    "pi_scarcity": (0.0, 1.5),
    "pi_novelty": (0.0, 1.5),
    "pi_spend": (0.0, 1.5),
    "accelerant_intensity": (0.0, 2.0),
    "mu_exposure": (0.0, 1.0),
    "nu_external": (0.0, 0.5),
    "lambda_fatigue": (0.0, 1.0),
    "zeta_volume": (0.0, 1.0),
    "alpha_quality": (0.5, 3.0),
    "beta_scarcity": (0.5, 3.0),
    "beta_coherence": (0.5, 3.0),
    "gamma_memory": (0.5, 3.0),
    "rho_novelty": (0.0, 2.0),
    "kappa_base": (0.0, 2.0),
    "kappa_trend": (0.0, 2.0),
    "psi_cost": (0.0, 2.0),
    "delta_distance": (0.0, 4.0),
    "delta_safe": (0.0, 4.0),
    "chi_premature": (0.0, 2.0),
    "omega_exposure": (0.0, 2.0),
    "eta_fatigue": (0.0, 5.0),
}


@dataclass
class YearningState:
    saturation: float = 0.0
    desire: float = 0.6
    affinity: float = 0.0  # m_x
    potential: float = 0.5  # p_x
    exposure: float = 0.2  # a_x
    fatigue: float = 0.0  # h

    def as_dict(self) -> Dict[str, float]:
        return {
            "saturation": float(self.saturation),
            "desire": float(self.desire),
            "affinity": float(self.affinity),
            "potential": float(self.potential),
            "exposure": float(self.exposure),
            "fatigue": float(self.fatigue),
        }


class SPrimeController:
    """Compute S′ engagement scores for controller actions."""

    def __init__(self, config: SPrimeConfig, bandit_weights: MutableMapping[str, Dict[str, float]]) -> None:
        self.config = config
        self.bandit_weights = bandit_weights
        self._yearnings: Dict[str, YearningState] = {}
        self._fatigue: float = 0.0
        self._reflection_boosts: Dict[str, float] = {}

    def score_action(
        self,
        action: ControllerAction,
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
        cooldown_time: int,
    ) -> Tuple[float, str]:
        cfg = self.config
        novelty = self._bounded(salience.get("novelty", 0.0))
        aim = self._bounded(salience.get("alignment", 0.0))
        key = self._bounded(salience.get("progress", 0.0))
        retention = self._bounded(1.0 - key)
        coherence = max(0.0, float(salience.get("coherence", 1.0)))
        drag = max(0.0, float(salience.get("drag", 0.0)))
        cost = max(0.0, float(salience.get("cost", 0.0)))

        novelty_factor = cfg.novelty_alpha * 4.0 * novelty * (1.0 - novelty) + cfg.novelty_beta * novelty
        base_value = (
            cfg.w_novelty * novelty_factor
            + cfg.w_retention * retention
            + cfg.w_payoff * aim
        )

        coherence_mult = coherence
        time_factor = 1.0 / (1.0 + cfg.discount_rate * max(0, cooldown_time))
        fatigue_factor = max(0.0, 1.0 - cfg.fatigue_k * drag) ** cfg.fatigue_gamma
        depth_factor = 1.0 + cfg.depth_factor_scale * float(action.cot_depth)

        s_prime = base_value * coherence_mult * time_factor * fatigue_factor * depth_factor
        s_prime -= cfg.cost_lambda * cost

        confidence = float(meta_snapshot.get("confidence", 0.0))
        roi = float(meta_snapshot.get("roi", 0.0))
        meta_boost = 1.0 + cfg.meta_confidence_boost * confidence + cfg.meta_roi_boost * roi
        s_prime *= meta_boost

        affinity_gain, scarcity_gain, accelerant_factor, fatigue_factor = self._dynamic_modifiers(action, salience)
        yearning_gain, yearning_state = self._yearning_gain(action)
        yearning_gain *= self._consume_reflection_boost(action)
        s_prime *= affinity_gain
        s_prime *= scarcity_gain
        s_prime *= yearning_gain
        s_prime *= accelerant_factor
        s_prime *= fatigue_factor

        bandit_bias = self._bandit_bias(action)
        s_prime += bandit_bias

        rationale = (
            f"novelty={novelty:.2f} novelty_factor={novelty_factor:.3f} retention={retention:.2f} "
            f"aim={aim:.2f} coherence={coherence_mult:.2f} time={time_factor:.2f} "
            f"fatigue={fatigue_factor:.2f} depth={depth_factor:.2f} cost_penalty={cfg.cost_lambda * cost:.3f} "
            f"meta_boost={meta_boost:.2f} affinity_gain={affinity_gain:.2f} scarcity_gain={scarcity_gain:.2f} "
            f"accelerant_factor={accelerant_factor:.2f} fatigue_factor={fatigue_factor:.2f} yearning_gain={yearning_gain:.2f} "
            f"yearning_S={yearning_state.saturation:.2f} yearning_D={yearning_state.desire:.2f} "
            f"bandit={bandit_bias:.3f}"
        )
        return s_prime, rationale

    def learn_from_outcome(self, action: ControllerAction, reward: float, lr: float = 0.05) -> None:
        key = self._action_key(action)
        bucket = self.bandit_weights.setdefault(key, {"bias": 0.0, "count": 0.0})
        bias = float(bucket.get("bias", 0.0))
        count = float(bucket.get("count", 0.0)) + 1.0
        bucket["bias"] = bias + lr * (reward - bias)
        bucket["count"] = count

    def register_actions(self, actions: Iterable[ControllerAction]) -> None:
        for action in actions:
            key = self._action_key(action)
            self._yearnings.setdefault(key, YearningState())
            self._reflection_boosts.setdefault(key, 0.0)

    def enqueue_reflection_boost(self, target_operator: ControllerOperator, boost: float) -> None:
        scaled = max(0.0, float(boost))
        if scaled <= 0.0:
            return
        for action_key in list(self._yearnings.keys()):
            if f"|op={target_operator.name}|" in action_key:
                self._reflection_boosts[action_key] = max(self._reflection_boosts.get(action_key, 0.0), scaled)

    def update_yearnings(self, chosen: ControllerAction, actions: Iterable[ControllerAction], salience: Mapping[str, float]) -> None:
        if not self.config.yearning_enabled:
            return
        cfg = self.config
        chosen_key = self._action_key(chosen)
        cfg_dyn = cfg.dynamics
        novelty = max(0.0, float(salience.get("novelty", 0.0)))
        presence_sum = 0.0
        for action in actions:
            key = self._action_key(action)
            state = self._yearnings.setdefault(key, YearningState())
            presence = 1.0 if key == chosen_key else 0.0
            presence_sum += presence
            state.saturation = (1.0 - cfg.yearning_rho) * state.saturation + cfg.yearning_sigma * presence
            state.desire = state.desire + cfg.yearning_alpha * (1.0 - state.desire) * (1.0 - presence)
            state.desire -= cfg.yearning_beta * state.desire * presence
            state.saturation = max(0.0, min(1.0, state.saturation))
            state.desire = max(0.0, min(1.0, state.desire))
            prev_exposure = state.exposure
            state.affinity = (1.0 - cfg_dyn.lambda_affinity) * state.affinity + cfg_dyn.xi_affinity * prev_exposure
            state.potential = state.potential + cfg_dyn.pi_scarcity * (1.0 - prev_exposure) + cfg_dyn.pi_novelty * novelty * (1.0 - presence) - cfg_dyn.pi_spend * cfg_dyn.accelerant_intensity * presence
            state.potential = max(0.0, min(1.5, state.potential))
            state.exposure = min(1.0, max(0.0, cfg_dyn.mu_exposure * prev_exposure + presence))
            state.fatigue = self._fatigue
        self._fatigue = max(0.0, cfg_dyn.lambda_fatigue * self._fatigue + cfg_dyn.zeta_volume * presence_sum)
        for state in self._yearnings.values():
            state.fatigue = self._fatigue

    def _bandit_bias(self, action: ControllerAction) -> float:
        bucket = self.bandit_weights.get(self._action_key(action))
        if not bucket:
            return 0.0
        return float(bucket.get("bias", 0.0))

    def _yearning_gain(self, action: ControllerAction) -> Tuple[float, YearningState]:
        cfg = self.config
        key = self._action_key(action)
        state = self._yearnings.setdefault(key, YearningState())
        if not cfg.yearning_enabled:
            return 1.0, state
        desire_minus_saturation = max(0.0, state.desire - state.saturation)
        gain = cfg.yearning_epsilon + desire_minus_saturation
        return gain, state

    def _dynamic_modifiers(self, action: ControllerAction, salience: Mapping[str, float]) -> Tuple[float, float, float, float]:
        cfg_dyn = self.config.dynamics
        state = self._yearnings.setdefault(self._action_key(action), YearningState())
        novelty = max(0.0, float(salience.get("novelty", 0.0)))
        trend_signal = float(salience.get(cfg_dyn.trend_key, 0.0))
        distance = float(salience.get(cfg_dyn.distance_key, cfg_dyn.delta_distance))
        q_val, c_val = self._quality_terms(action, salience)
        quality_term = (max(cfg_dyn.epsilon, q_val) ** cfg_dyn.alpha_quality) * (max(cfg_dyn.epsilon, c_val) ** cfg_dyn.beta_coherence)
        memory_term = max(cfg_dyn.epsilon, state.affinity * max(0.0, 1.0 - state.exposure)) ** cfg_dyn.gamma_memory
        affinity_gain = quality_term * memory_term * (1.0 + cfg_dyn.rho_novelty * novelty)
        scarcity_gain = max(cfg_dyn.epsilon, state.potential) ** cfg_dyn.beta_scarcity
        trend_gate = 1.0 if trend_signal < 0.0 else 0.0
        accelerant = 1.0 + (cfg_dyn.kappa_base + cfg_dyn.kappa_trend * trend_gate) * cfg_dyn.accelerant_intensity
        accelerant -= cfg_dyn.psi_cost * (cfg_dyn.accelerant_intensity ** 2) / max(cfg_dyn.epsilon, state.potential + cfg_dyn.epsilon)
        accelerant = max(cfg_dyn.epsilon, accelerant)
        exposure_penalty = max(cfg_dyn.epsilon, 1.0 - cfg_dyn.omega_exposure * state.exposure)
        distance_penalty = max(cfg_dyn.epsilon, 1.0 - cfg_dyn.chi_premature * max(0.0, cfg_dyn.delta_safe - distance))
        accelerant_factor = accelerant * exposure_penalty * distance_penalty
        fatigue_effect = math.exp(-cfg_dyn.eta_fatigue * self._fatigue)
        return affinity_gain, scarcity_gain, accelerant_factor, fatigue_effect

    def _quality_terms(self, action: ControllerAction, salience: Mapping[str, float]) -> Tuple[float, float]:
        coherence = max(0.0, min(1.0, float(salience.get("coherence", 0.6))))
        base_quality = {
            "SASS": 0.65,
            "SASS_WITH_JUMP": 0.8,
            "MEMORY_OP": 0.6,
            "TOOL": 0.62,
            "VERIFY": 0.7,
            "REFLECT": 0.78,
        }.get(action.operator.name, 0.6)
        depth_bonus = 0.05 * min(action.cot_depth, 3)
        quality = max(0.0, min(1.0, base_quality + depth_bonus))
        return quality, coherence

    # ------------------------------------------------------------------
    # Introspection & safe self-adjustment
    # ------------------------------------------------------------------
    def snapshot_yearnings(self) -> Dict[str, Dict[str, float]]:
        return {key: state.as_dict() for key, state in self._yearnings.items()}

    def _consume_reflection_boost(self, action: ControllerAction) -> float:
        key = self._action_key(action)
        boost = self._reflection_boosts.get(key, 0.0)
        if boost <= 0.0:
            return 1.0
        self._reflection_boosts[key] = 0.0
        return 1.0 + boost

    def guarded_adjust_dynamics(
        self,
        updates: Mapping[str, float],
        *,
        max_step: float = 0.1,
    ) -> Dict[str, float]:
        applied: Dict[str, float] = {}
        cfg_dyn = self.config.dynamics
        for key, delta in updates.items():
            if key not in DYNAMICS_FIELD_BOUNDS:
                continue
            bounds = DYNAMICS_FIELD_BOUNDS[key]
            current = getattr(cfg_dyn, key, None)
            if not isinstance(current, (int, float)):
                continue
            clamped_delta = max(-max_step, min(max_step, float(delta)))
            new_value = self._clamp(current + clamped_delta, bounds[0], bounds[1])
            setattr(cfg_dyn, key, new_value)
            applied[key] = new_value
        return applied

    @staticmethod
    def _action_key(action: ControllerAction) -> str:
        return f"depth={action.cot_depth}|op={action.operator.name}|patch={action.patch.name}"

    @staticmethod
    def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
        if math.isnan(value):
            return 0.0
        if value < low:
            return low
        if value > high:
            return high
        return value
