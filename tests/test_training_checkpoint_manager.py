from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from salience_os_seed.proto_lm.checkpoints import CheckpointManager
from salience_os_seed.training.checkpoint_manager import (
    CheckpointNotification,
    TrainingCheckpointManager,
)


def _dummy_payload(step: int) -> dict[str, object]:
    return {
        "model": {"weight": torch.tensor(float(step))},
        "optimizer": {},
        "step": step,
        "vocab": {"tokens": ["a", "b"], "merges": []},
    }


def test_training_checkpoint_manager_registers_remote_checkpoint(tmp_path: Path) -> None:
    active_path = tmp_path / "active.pt"
    base = CheckpointManager(tmp_path / "repo", active_path=active_path)
    manager = TrainingCheckpointManager(base)

    payload_path = tmp_path / "remote.pt"
    torch.save(_dummy_payload(5), payload_path)
    metadata_path = tmp_path / "remote.json"
    metadata_path.write_text(
        json.dumps({"step": 5, "reason": "eval", "tags": ["candidate"], "metrics": {"loss": 0.01}}),
        encoding="utf-8",
    )

    notifications: List[CheckpointNotification] = []
    manager.subscribe(lambda note: notifications.append(note))

    record = manager.register_mcp_checkpoint(
        {
            "checkpoint_path": str(payload_path),
            "metadata_path": str(metadata_path),
            "job_id": "job-123",
            "verdict": "baseline",
        }
    )

    assert record is not None
    assert record.metadata["status"] == "promoted"
    assert base.active_record() is not None
    assert notifications and notifications[0].status == "promoted"
    extra = record.metadata.get("extra", {})
    assert isinstance(extra, dict)
    assert extra.get("origin") == "mcp"
    assert extra.get("job_id") == "job-123"


def test_training_checkpoint_manager_promotes_existing_identifier(tmp_path: Path) -> None:
    active_path = tmp_path / "active.pt"
    base = CheckpointManager(tmp_path / "repo", active_path=active_path)
    existing = base.create_checkpoint(_dummy_payload(3), step=3, reason="seed")
    manager = TrainingCheckpointManager(base)

    notifications: List[CheckpointNotification] = []
    manager.subscribe(lambda note: notifications.append(note))

    record = manager.register_mcp_checkpoint({"identifier": existing.identifier, "verdict": "accept"})
    assert record is not None
    assert record.metadata["status"] == "promoted"
    assert base.active_record() is not None
    assert notifications and notifications[0].status == "promoted"
