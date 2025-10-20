import numpy as np

from salience_os_seed.core.memory import StructuredMemory
from salience_os_seed.core.sensors import SensorBank


def test_sensor_bank_emits_normalised_vector():
    bank = SensorBank.default_bank()
    memory = StructuredMemory()
    memory_snapshot = memory.as_runtime_mapping()

    logits = np.array([
        [1.2, 0.7, -0.1, 0.3],
        [1.1, 0.6, -0.2, 0.2],
    ])
    state = {
        "prediction": {
            "token_logits": logits,
            "steps_remaining": 5.0,
            "entropy_estimate": 1.8,
        },
        "context": {
            "tokens": ["alpha", "beta", "gamma", "delta"],
            "text": "alpha beta gamma delta",
        },
        "decision_proposal": {
            "operator": "SASS",
            "cot_depth": 1,
        },
        "prompt": "alpha beta gamma",
        "last_action": "REFLECT",
        "scratchpad_text": "considering alpha and beta relationships",
        "contradictions": 0.0,
    }
    meta_snapshot = {
        "confidence": 0.2,
        "difficulty": 0.5,
        "roi": 0.3,
    }

    salience_vector = bank.tick(state, memory_snapshot, meta_snapshot)

    assert salience_vector.values.shape[0] == 7
    assert salience_vector.values.dtype == np.float32
    assert all(np.isfinite(salience_vector.values))
    assert set(salience_vector.as_mapping().keys()) == {
        "uncertainty",
        "novelty",
        "alignment",
        "progress",
        "cost",
        "drag",
        "coherence",
    }
