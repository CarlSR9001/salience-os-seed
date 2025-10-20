"""Checkpoint utilities for integrating MCP-produced artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Mapping, MutableSequence, Optional, Sequence

import torch

from ..proto_lm.checkpoints import CheckpointManager, CheckpointRecord
from ..telemetry import BUS, TelemetryEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointNotification:
    """Notification describing a checkpoint update relevant to inference workers."""

    record: CheckpointRecord
    status: str
    origin: str
    metadata: Mapping[str, object]


class TrainingCheckpointManager:
    """Register MCP-produced checkpoints and notify inference subscribers."""

    def __init__(
        self,
        manager: CheckpointManager,
        *,
        origin: str = "mcp",
        notify_bus: bool = True,
        subscribers: Iterable[Callable[[CheckpointNotification], None]] | None = None,
    ) -> None:
        self.manager = manager
        self.origin = origin
        self._notify_bus = notify_bus
        self._subscribers: MutableSequence[Callable[[CheckpointNotification], None]] = list(subscribers or [])
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------
    def subscribe(self, callback: Callable[[CheckpointNotification], None]) -> Callable[[], None]:
        """Subscribe to checkpoint notifications."""

        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return _unsubscribe

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_mcp_checkpoint(
        self,
        descriptor: Mapping[str, object],
        *,
        promote: Optional[bool] = None,
        reason: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> CheckpointRecord | None:
        """Register a checkpoint produced by an MCP job."""

        identifier = descriptor.get("identifier")
        if isinstance(identifier, str):
            existing = self.manager.get(identifier)
            if existing is not None:
                record = existing
                if promote is not False:
                    updated = self.manager.promote(identifier, verdict=str(descriptor.get("verdict", "mcp")))
                    if updated is not None:
                        record = updated
                self._notify(record, status="promoted" if promote is not False else "registered")
                return record

        path_value = descriptor.get("checkpoint_path") or descriptor.get("payload_path")
        if not isinstance(path_value, str):
            logger.warning("Cannot register MCP checkpoint without payload path: %r", descriptor)
            return None
        payload_path = Path(path_value).expanduser()
        if not payload_path.exists():
            logger.warning("MCP checkpoint path does not exist: %s", payload_path)
            return None

        metadata = self._load_descriptor_metadata(descriptor)
        try:
            payload = torch.load(payload_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load checkpoint payload from %s: %s", payload_path, exc)
            return None
        if not isinstance(payload, Mapping):
            logger.warning("Checkpoint payload %s is not a mapping", payload_path)
            return None

        step = self._resolve_step(descriptor, metadata, payload)
        applied_reason = reason or metadata.get("reason") or "mcp-import"
        tag_list = list(tags or metadata.get("tags", []))
        if "mcp" not in tag_list:
            tag_list.append("mcp")
        extra_metadata = dict(metadata)
        extra_metadata.setdefault("origin", self.origin)
        job_id = descriptor.get("job_id")
        if isinstance(job_id, str):
            extra_metadata.setdefault("job_id", job_id)

        record = self.manager.create_checkpoint(
            payload,
            step=step,
            reason=applied_reason,
            tags=tag_list,
            extra_metadata=extra_metadata,
            auto_evaluate=False,
        )

        final_record = record
        if promote is not False:
            promoted = self.manager.promote(record.identifier, verdict=str(descriptor.get("verdict", "mcp")))
            if promoted is not None:
                final_record = promoted
                self._notify(final_record, status="promoted")
                return final_record

        self._notify(final_record, status="registered")
        return final_record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_descriptor_metadata(self, descriptor: Mapping[str, object]) -> Mapping[str, object]:
        metadata = descriptor.get("metadata")
        if isinstance(metadata, Mapping):
            return dict(metadata)
        path_value = descriptor.get("metadata_path")
        if isinstance(path_value, str):
            path = Path(path_value).expanduser()
            try:
                with path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except (OSError, json.JSONDecodeError):
                return {}
            if isinstance(loaded, Mapping):
                return dict(loaded)
        return {}

    def _resolve_step(
        self,
        descriptor: Mapping[str, object],
        metadata: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> int:
        for source in (
            descriptor.get("step"),
            metadata.get("step"),
            payload.get("step"),
        ):
            try:
                if source is None:
                    continue
                return int(source)
            except (TypeError, ValueError):
                continue
        return 0

    def _notify(self, record: CheckpointRecord, *, status: str) -> None:
        payload = {
            "identifier": record.identifier,
            "status": status,
            "path": str(record.payload_path),
            "metadata_path": str(record.metadata_path),
            "origin": self.origin,
            "step": record.metadata.get("step"),
        }
        notification = CheckpointNotification(
            record=record,
            status=status,
            origin=self.origin,
            metadata=record.metadata,
        )
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(notification)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Inference subscriber failed for checkpoint %s", record.identifier)
        if self._notify_bus:
            BUS.publish(TelemetryEvent(type="training/checkpoint", payload=payload))


__all__ = ["TrainingCheckpointManager", "CheckpointNotification"]
