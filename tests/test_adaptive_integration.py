import json
from pathlib import Path

import pytest

from salience_os_seed.adaptive.manager import AdaptiveCoordinator
from salience_os_seed.conversation.session import ConversationConfig, ConversationSession
from salience_os_seed.core.controller.actions import ControllerAction, ControllerOperator, ControllerPatch
from salience_os_seed.core.controller.policy import ControllerDecision
from salience_os_seed.proto_lm.trainer import ProtoLanguageModel, TrainingConfig
from salience_os_seed.runtime.orchestrator import RuntimeConfig, RuntimeMetrics, SalienceRuntime


def _build_runtime_metrics(step: int) -> RuntimeMetrics:
    action = ControllerAction(cot_depth=1, operator=ControllerOperator.SASS, patch=ControllerPatch.NONE)
    decision = ControllerDecision(
        action=action,
        score=1.25,
        salience_mapping={
            "novelty": 0.4,
            "progress": 0.5,
            "roi": 0.6,
            "drag": 0.05,
            "cost": 0.02,
            "alignment": 0.7,
            "truth": 0.75,
            "uncertainty": 0.1,
            "risk": 0.05,
        },
        cooldown_steps=0,
        hysteresis_delta=0.2,
    )
    metrics = RuntimeMetrics(
        step=step,
        decision=decision,
        meta_report="",
        verification_passed=True,
        budget_left=500.0,
        scheduler_snapshot={},
        idea_acceptances=0,
        yearning_snapshot={},
        maintenance_report={},
        experiment_reports=tuple(),
        episode_recorded=None,
    )
    return metrics


def test_adaptive_coordinator_roundtrip():
    runtime = SalienceRuntime(RuntimeConfig(budget_tokens=256))
    proto = ProtoLanguageModel(TrainingConfig(learning_rate=1e-4))
    coordinator = AdaptiveCoordinator(runtime=runtime, proto_lm=proto)

    snapshot = {
        "loss": 1.2,
        "loss_components": {"old": 1.4, "new": 1.0},
        "grad_health": {"grad_norm": 0.8, "frac_nonzero": 0.9},
    }
    coordinator.observe_training(snapshot)
    metrics = _build_runtime_metrics(step=1)
    coordinator.track_runtime(metrics)
    response, summary = coordinator.assess_response("sample reply", metrics)
    assert isinstance(response, str)
    assert summary.truth_decision.decision in {"DROP", "SAVE", "SPEAK"}

    exported = coordinator.export_state()
    # Ensure the export is json-serialisable for persistence.
    json.loads(json.dumps(exported))

    runtime_restored = SalienceRuntime(RuntimeConfig(budget_tokens=256))
    proto_restored = ProtoLanguageModel(TrainingConfig(learning_rate=1e-4))
    restored = AdaptiveCoordinator(runtime=runtime_restored, proto_lm=proto_restored)
    restored.import_state(exported)
    assert restored.last_gating_summary is not None


def test_conversation_session_persists_adaptive_state(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    lm_config = TrainingConfig(checkpoint_path=str(checkpoint_path))
    config = ConversationConfig(lm=lm_config, learning_enabled=False)

    session = ConversationSession(config=config)
    session.process_user_input("Hello there")
    snapshot = session.generate_response()
    assert snapshot.response
    state_path = tmp_path / "session_state.json"
    session.save_state(state_path)
    assert state_path.exists()

    # Create a fresh session and load the saved state.
    session_loaded = ConversationSession(config=config)
    session_loaded.load_state(state_path)
    assert session_loaded.history
    assert session_loaded._adaptive.last_gating_summary is not None
    assert session_loaded.proto_lm._external_state_exporter is not None
