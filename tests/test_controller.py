import math

from salience_os_seed.core.controller import (
    ControllerAction,
    ControllerOperator,
    ControllerPatch,
    SalienceControllerPolicy,
)
from salience_os_seed.core.controller.policy import ControllerConfig


def test_controller_hysteresis_respects_threshold():
    actions = [
        ControllerAction(cot_depth=0, operator=ControllerOperator.SASS, patch=ControllerPatch.NONE),
        ControllerAction(cot_depth=1, operator=ControllerOperator.SASS_WITH_JUMP, patch=ControllerPatch.NONE),
    ]
    config = ControllerConfig(
        delta_weight=0.0,
        aim_weight=0.0,
        key_weight=0.0,
        drag_penalty_strength=0.0,
        drag_gamma=1.0,
        lambda_cost=0.0,
        hysteresis_threshold=0.5,
        cooldown_steps=1,
    )
    bandit_store = {}
    policy = SalienceControllerPolicy(config=config, bandit_weights=bandit_store, available_actions=actions)

    base_salience = {
        "uncertainty": 1.0,
        "novelty": 0.0,
        "alignment": 0.0,
        "progress": 0.0,
        "cost": 0.0,
        "drag": 0.0,
    }
    meta_snapshot = {"confidence": 0.0, "roi": 0.0}
    decision1 = policy.choose(base_salience, meta_snapshot, budget_left=1.0)
    assert decision1.action == actions[1]
    first_score = decision1.score

    key_action0 = SalienceControllerPolicy._action_key(actions[0])
    key_action1 = SalienceControllerPolicy._action_key(actions[1])
    policy.bandit_weights[key_action0] = {"bias": 0.35, "count": 1.0}
    policy.bandit_weights[key_action1] = {"bias": 0.0, "count": 1.0}

    adjusted_salience = dict(base_salience)
    adjusted_salience["uncertainty"] = 0.6
    candidate_score, _ = policy._score_action(actions[0], adjusted_salience, meta_snapshot, budget_left=1.0)
    other_score, _ = policy._score_action(actions[1], adjusted_salience, meta_snapshot, budget_left=1.0)
    assert candidate_score > other_score
    delta = candidate_score - policy.state.last_score
    assert delta < config.hysteresis_threshold

    decision2 = policy.choose(adjusted_salience, meta_snapshot, budget_left=1.0)
    assert decision2.action == actions[1]
    assert math.isclose(policy.state.last_score, first_score, rel_tol=1e-5)

    policy.bandit_weights[key_action0]["bias"] = 1.2
    decision3 = policy.choose(adjusted_salience, meta_snapshot, budget_left=1.0)
    assert decision3.action == actions[0]
    assert decision3.hysteresis_delta > config.hysteresis_threshold
