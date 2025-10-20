"""Bandit-driven salience controller policy.

The policy consumes salience vectors (normalised sensor outputs) plus meta-state
signals and selects an action tuple (`cot_depth`, `operator`, `patch`).

Key behaviours implemented:
- Score computation following the architectural spec. The controller maintains
  per-action weights tuned by the `BanditTrainer`.
- Hysteresis: the controller only flips to a new action if its score exceeds the
  incumbent by `delta_threshold`.
- Cooldown: after switching actions we enforce a minimum number of steps before
  another change is allowed.
- Compute auction hooks: operators submit bids (expected gain − λ·cost). The
  policy combines bandit scores with auction bids when available.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .actions import ControllerAction, ControllerDecision, ControllerOperator, ControllerPatch
from .s_prime import SPrimeConfig, SPrimeController


@dataclass
class ControllerConfig:
    """Configuration bundle for the controller policy."""

    delta_weight: float = 1.0  # w1
    aim_weight: float = 1.2  # w2
    key_weight: float = 1.4  # w3
    drag_penalty_strength: float = 0.4  # k
    drag_gamma: float = 1.3
    lambda_cost: float = 0.015
    cooldown_steps: int = 2
    hysteresis_threshold: float = 0.6
    base_gain_scale: float = 1.0
    score_floor: float = -5.0
    score_ceiling: float = 8.0


@dataclass
class ControllerState:
    """Mutable runtime state tracked by the policy."""

    last_action: Optional[ControllerAction] = None
    last_score: float = 0.0
    cooldown_remaining: int = 0
    steps: int = 0


class SalienceControllerPolicy:
    """Implements the salience-aware action selection routine."""

    _LOGGER = logging.getLogger(__name__)

    def __init__(
        self,
        config: ControllerConfig,
        bandit_weights: MutableMapping[str, Dict[str, float]],
        available_actions: Optional[Sequence[ControllerAction]] = None,
        s_prime_config: Optional[SPrimeConfig] = None,
    ) -> None:
        self.config = config
        self.state = ControllerState()
        self.bandit_weights = bandit_weights
        self.actions = list(available_actions or self._default_action_space())
        self._auction_bids: Dict[ControllerAction, float] = {}
        self._last_score_components: Dict[ControllerAction, Tuple[float, str]] = {}
        self.s_prime = SPrimeController(s_prime_config or SPrimeConfig(), bandit_weights)
        self.s_prime.register_actions(self.actions)

    def register_auction_bid(self, action: ControllerAction, bid: float) -> None:
        """Operators can register auction bids before `choose` is called."""

        self._auction_bids[action] = bid

    def reset_auction_bids(self) -> None:
        self._auction_bids.clear()

    def choose(
        self,
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
        budget_left: float,
    ) -> ControllerDecision:
        """Compute scores for all actions and return the best candidate."""

        cfg = self.config
        previous_cooldown = self.state.cooldown_remaining
        scores: Dict[ControllerAction, float] = {}
        details: Dict[ControllerAction, Tuple[float, str]] = {}
        for action in self.actions:
            score, rationale = self._score_action(action, salience, meta_snapshot, budget_left)
            if action in self._auction_bids:
                score += self._auction_bids[action]
            score = float(np.clip(score, cfg.score_floor, cfg.score_ceiling))
            scores[action] = score
            details[action] = (score, rationale)

        chosen, score, hysteresis_delta = self._apply_hysteresis(scores)
        cooldown = max(self.state.cooldown_remaining - 1, 0)
        self.state.steps += 1
        if chosen != self.state.last_action:
            cooldown = cfg.cooldown_steps
        self.state.last_action = chosen
        self.state.last_score = score
        self.state.cooldown_remaining = cooldown
        self._last_score_components = details
        self.s_prime.update_yearnings(chosen, self.actions, salience)
        decision = ControllerDecision(
            action=chosen,
            score=score,
            salience_mapping=dict(salience),
            cooldown_steps=cooldown,
            hysteresis_delta=hysteresis_delta,
        )
        self._log_decision_trace(
            salience,
            decision,
            scores,
            details,
            previous_cooldown=previous_cooldown,
        )
        self.reset_auction_bids()
        return decision

    def _score_action(
        self,
        action: ControllerAction,
        salience: Mapping[str, float],
        meta_snapshot: Mapping[str, float],
        budget_left: float,
    ) -> Tuple[float, str]:
        cooldown_time = max(self.state.cooldown_remaining, 0)
        s_prime_score, s_prime_rationale = self.s_prime.score_action(
            action,
            salience,
            meta_snapshot,
            cooldown_time=cooldown_time,
        )
        budget_factor = 1.0 if budget_left <= 0 else float(np.tanh(budget_left))
        final_score = s_prime_score * budget_factor
        rationale = f"S'={s_prime_score:.3f} budget={budget_factor:.3f} | {s_prime_rationale}"
        return final_score, rationale

    @staticmethod
    def _action_summary(action: ControllerAction) -> Mapping[str, object]:
        return {
            "operator": action.operator.name,
            "cot_depth": action.cot_depth,
            "patch": action.patch.name,
        }

    def _log_decision_trace(
        self,
        salience: Mapping[str, float],
        decision: ControllerDecision,
        scores: Mapping[ControllerAction, float],
        details: Mapping[ControllerAction, Tuple[float, str]],
        *,
        previous_cooldown: int,
    ) -> None:
        if not self._LOGGER.isEnabledFor(logging.INFO):
            return
        bid_winners = sorted(
            ((self._action_key(action), float(bid)) for action, bid in self._auction_bids.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_bids = [
            {"action": action_key, "score": bid}
            for action_key, bid in bid_winners[:5]
        ]
        score_trace = {
            self._action_key(action): {
                "score": float(scores.get(action, 0.0)),
                "rationale": rationale,
            }
            for action, (_, rationale) in details.items()
        }
        payload = {
            "salience": {name: float(value) for name, value in salience.items()},
            "decision": {
                "action": self._action_summary(decision.action),
                "score": float(decision.score),
                "hysteresis_delta": float(decision.hysteresis_delta),
            },
            "cooldown": {
                "previous": int(previous_cooldown),
                "next": int(decision.cooldown_steps),
            },
            "auction_bids": top_bids,
            "score_trace": score_trace,
        }
        self._LOGGER.info(
            "controller.select_action",
            extra={"controller_decision": payload},
        )

    def _bandit_score(self, action: ControllerAction) -> float:
        action_key = self._action_key(action)
        weights = self.bandit_weights.get(action_key)
        if not weights:
            return 0.0
        return float(weights.get("bias", 0.0))

    def notify_outcome(
        self,
        action: ControllerAction,
        reward: float,
        learning_rate: float,
    ) -> None:
        action_key = self._action_key(action)
        bucket = self.bandit_weights.setdefault(action_key, {"bias": 0.0, "count": 0.0})
        count = bucket.get("count", 0.0) + 1.0
        bias = bucket.get("bias", 0.0)
        bucket["bias"] = bias + learning_rate * (reward - bias)
        bucket["count"] = count

    def _apply_hysteresis(self, scores: Mapping[ControllerAction, float]) -> tuple[ControllerAction, float, float]:
        cfg = self.config
        if not scores:
            raise ValueError("No scores provided to controller")
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_action, best_score = items[0]
        if (
            self.state.cooldown_remaining > 0
            and self.state.last_action is not None
            and best_action != self.state.last_action
        ):
            return self.state.last_action, self.state.last_score, 0.0
        if self.state.last_action is None:
            return best_action, best_score, float("inf")
        prev_score = self.state.last_score
        delta = best_score - prev_score
        if best_action == self.state.last_action or delta > cfg.hysteresis_threshold:
            return best_action, best_score, delta
        return self.state.last_action, prev_score, delta

    @staticmethod
    def _action_key(action: ControllerAction) -> str:
        return f"depth={action.cot_depth}|op={action.operator.name}|patch={action.patch.name}"

    @staticmethod
    def _default_action_space() -> Sequence[ControllerAction]:
        depths = [0, 1, 3]
        operators = [
            ControllerOperator.SASS,
            ControllerOperator.SASS_WITH_JUMP,
            ControllerOperator.MEMORY_OP,
            ControllerOperator.TOOL,
            ControllerOperator.VERIFY,
            ControllerOperator.REFLECT,
        ]
        patches = [ControllerPatch.NONE, ControllerPatch.MATH, ControllerPatch.RETRIEVAL, ControllerPatch.PLAN]
        actions = []
        for depth in depths:
            for operator in operators:
                for patch in patches:
                    if patch is not ControllerPatch.NONE and operator in {
                        ControllerOperator.MEMORY_OP,
                        ControllerOperator.TOOL,
                        ControllerOperator.VERIFY,
                    }:
                        # skip nonsensical combos
                        continue
                    actions.append(ControllerAction(cot_depth=depth, operator=operator, patch=patch))
        return actions

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def last_scores(self) -> Sequence[Tuple[ControllerAction, float, str]]:
        return [
            (action, score, rationale)
            for action, (score, rationale) in self._last_score_components.items()
        ]

    def policy_diagnostics(self) -> Dict[str, object]:
        return {
            "steps": self.state.steps,
            "cooldown_remaining": self.state.cooldown_remaining,
            "last_action": self.state.last_action,
            "last_score": self.state.last_score,
            "hysteresis_threshold": self.config.hysteresis_threshold,
            "actions": len(self.actions),
        }

    # ------------------------------------------------------------------
    # External nudges
    # ------------------------------------------------------------------
    def enqueue_reflection(self, boost: float = 1.0) -> None:
        """Schedule a reflection action by boosting its yearning temporarily."""

        target = ControllerOperator.REFLECT
        scaled_boost = max(0.0, float(boost))
        if scaled_boost <= 0.0:
            return
        self.s_prime.enqueue_reflection_boost(target, scaled_boost)
