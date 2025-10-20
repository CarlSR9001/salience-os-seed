import pytest

torch = pytest.importorskip("torch")

from salience_os_seed.core.controller import (
    ControllerAction,
    ControllerDecision,
    ControllerOperator,
    ControllerPatch,
)
from salience_os_seed.runtime.orchestrator import SalienceRuntime


def _make_sass_decision() -> ControllerDecision:
    action = ControllerAction(
        cot_depth=0,
        operator=ControllerOperator.SASS,
        patch=ControllerPatch.NONE,
    )
    return ControllerDecision(
        action=action,
        score=0.0,
        salience_mapping={},
        cooldown_steps=0,
        hysteresis_delta=0.0,
    )


def test_training_step_preserves_gradients_across_invocations():
    runtime = SalienceRuntime()
    runtime.set_training_active(True)

    hidden = torch.randn(1, 3, runtime.config.sass.d_model, requires_grad=True)
    state = {"hidden_states": hidden, "training_active": True}
    decision = _make_sass_decision()

    runtime._execute_action(decision, state, {})
    runtime._execute_action(decision, {"training_active": True}, {})

    loss = runtime.hidden_states.sum()
    loss.backward()

    assert hidden.grad is not None
    assert torch.any(hidden.grad != 0)
