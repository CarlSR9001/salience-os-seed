import math

import pytest

torch = pytest.importorskip("torch")

from salience_os_seed.proto_lm.trainer import ProtoLanguageModel, TrainingConfig  # noqa: E402


def test_vocab_expansion_preserves_optimizer_state():
    config = TrainingConfig(
        embed_dim=16,
        learning_rate=5e-4,
        vocab_growth_chunk=4,
        vocab_growth_headroom=0,
        vocab_lr_cooldown_steps=0,
        vocab_lr_multiplier=1.0,
        device="cpu",
    )
    proto = ProtoLanguageModel(config)
    proto.train()

    baseline_text = "hello world"
    loss_before = proto.training_step(baseline_text)
    assert math.isfinite(loss_before)

    embed_param = proto.embed.weight
    state = proto.optimizer.state[embed_param]
    exp_avg_before = state["exp_avg"].detach().clone()
    exp_avg_sq_before = state["exp_avg_sq"].detach().clone()

    expansion_text = "😀a😀"
    proto.encode(expansion_text)

    new_state = proto.optimizer.state[proto.embed.weight]
    assert new_state["exp_avg"].shape[0] > exp_avg_before.shape[0]

    preserved_rows = exp_avg_before.shape[0]
    torch.testing.assert_close(new_state["exp_avg"][:preserved_rows], exp_avg_before)
    torch.testing.assert_close(new_state["exp_avg_sq"][:preserved_rows], exp_avg_sq_before)
    assert torch.count_nonzero(new_state["exp_avg"][preserved_rows:]) == 0
    assert torch.count_nonzero(new_state["exp_avg_sq"][preserved_rows:]) == 0

    loss_after = proto.training_step(expansion_text)
    assert math.isfinite(loss_after)

