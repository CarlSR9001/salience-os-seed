import pytest

pytest.importorskip("torch")
import torch

from core.operators.sass import SASSConfig, SASSCore


def _build_core() -> SASSCore:
    config = SASSConfig(
        d_model=16,
        state_channels=16,
        kernel_size=3,
        num_layers=2,
        dropout=0.0,
    )
    return SASSCore(config)


def test_sasscore_retains_gradients_during_training():
    torch.manual_seed(0)
    core = _build_core()
    core.train()

    inputs = torch.randn(2, 3, core.config.d_model, requires_grad=True)
    outputs, states = core(inputs)

    assert any(state.requires_grad for state in states)

    loss = outputs.sum()
    loss.backward()

    assert inputs.grad is not None


def test_sasscore_can_detach_states_for_inference():
    torch.manual_seed(0)
    core = _build_core()
    core.eval()

    inputs = torch.randn(2, 3, core.config.d_model, requires_grad=True)
    outputs, states = core(inputs, detach_states=True)

    assert all(not state.requires_grad for state in states)

    loss = outputs.sum()
    loss.backward()

    assert inputs.grad is not None

    next_inputs = torch.randn(2, 1, core.config.d_model, requires_grad=True)
    _, next_states = core(next_inputs, layer_states=states, detach_states=True)

    assert all(not state.requires_grad for state in next_states)
