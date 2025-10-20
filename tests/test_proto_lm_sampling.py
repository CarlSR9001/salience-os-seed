import copy

import pytest

torch = pytest.importorskip("torch")

from salience_os_seed.proto_lm.trainer import ProtoLanguageModel, TrainingConfig


def _snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[int, dict[str, object]]:
    snapshot: dict[int, dict[str, object]] = {}
    for param, state in optimizer.state.items():
        param_state: dict[str, object] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                param_state[key] = value.clone()
            else:
                param_state[key] = copy.deepcopy(value)
        snapshot[id(param)] = param_state
    return snapshot


def _optimizer_states_equal(
    left: dict[int, dict[str, object]], right: dict[int, dict[str, object]]
) -> bool:
    if left.keys() != right.keys():
        return False
    for param_id, left_state in left.items():
        right_state = right.get(param_id)
        if right_state is None or left_state.keys() != right_state.keys():
            return False
        for key, left_value in left_state.items():
            right_value = right_state[key]
            if torch.is_tensor(left_value):
                if not torch.is_tensor(right_value) or not torch.equal(left_value, right_value):
                    return False
            else:
                if left_value != right_value:
                    return False
    return True


def test_sampling_does_not_mutate_vocab_or_optimizer_state():
    config = TrainingConfig(
        sequence_length=8,
        embed_dim=16,
        vocab_growth_chunk=16,
        vocab_growth_headroom=0,
    )
    model = ProtoLanguageModel(config=config, learning_enabled=False)
    prefix = "Hello world"

    initial_vocab_size = model.vocab.size()
    optimizer_state_before = _snapshot_optimizer_state(model.optimizer)

    for _ in range(3):
        model.sample(prefix, max_tokens=4)

    assert model.vocab.size() == initial_vocab_size
    optimizer_state_after = _snapshot_optimizer_state(model.optimizer)
    assert _optimizer_states_equal(optimizer_state_before, optimizer_state_after)
