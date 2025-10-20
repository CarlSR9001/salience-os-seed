from __future__ import annotations

import json
from pathlib import Path

import torch

from salience_os_seed.proto_lm.checkpoints import CheckpointManager
from salience_os_seed.telemetry import ParameterEvent, render_parameter_event


def _dummy_payload(step: int) -> dict[str, object]:
    return {
        "model": {"weight": torch.tensor(float(step))},
        "optimizer": {},
        "step": step,
        "vocab": {"tokens": ["a", "b"], "merges": []},
    }


def test_checkpoint_manager_promote_and_revert(tmp_path: Path) -> None:
    active_path = tmp_path / "active.pt"
    manager = CheckpointManager(tmp_path, active_path=active_path)

    first = manager.create_checkpoint(_dummy_payload(1), step=1, reason="initial")
    assert first.metadata["status"] == "candidate"
    promoted = manager.promote(first.identifier, verdict="baseline")
    assert promoted is not None
    assert active_path.exists()
    assert manager.active_record() is not None

    second = manager.create_checkpoint(_dummy_payload(2), step=2, reason="update")
    manager.promote(second.identifier, verdict="candidate")
    assert manager.active_record().identifier == second.identifier
    reverted = manager.revert_to(first.identifier, reason="regression")
    assert reverted is not None
    assert manager.active_record().identifier == first.identifier
    metadata = json.loads(reverted.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "promoted"
    assert metadata.get("reinstated_reason") == "regression"


def test_checkpoint_manager_resolve_load_path(tmp_path: Path) -> None:
    active_path = tmp_path / "active.pt"
    manager = CheckpointManager(tmp_path, active_path=active_path)
    record = manager.create_checkpoint(_dummy_payload(3), step=3, reason="snap", auto_evaluate=False)
    manager.promote(record.identifier, verdict="snap")
    resolved = manager.resolve_load_path()
    assert resolved is not None
    assert resolved.exists()
    matched = manager.find_by_payload(resolved)
    assert matched is not None


def test_gradient_event_rendering() -> None:
    event = ParameterEvent(payload={"step": 42, "total": 1000, "first_norm": 3.5}, kind="gradient")
    rendered = render_parameter_event(event)
    assert "grad_norm" in rendered
    assert "step=42" in rendered
