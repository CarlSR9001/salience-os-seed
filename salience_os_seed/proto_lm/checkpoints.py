"""Metadata-aware checkpoint management for ProtoLanguageModel."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence
from uuid import uuid4

import torch

try:
    from ..runtime.action_executor import MCPToolSession
except Exception:  # pragma: no cover - optional dependency during import cycles
    MCPToolSession = None  # type: ignore


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata describing a single checkpoint snapshot."""

    identifier: str
    payload_path: Path
    metadata_path: Path
    metadata: Mapping[str, object]


class CheckpointManager:
    """Coordinate checkpoint creation, promotion, and rollbacks."""

    def __init__(
        self,
        root: Path,
        *,
        active_path: Optional[Path] = None,
        evaluation_tool: Optional[str] = None,
        evaluation_suite: Optional[str] = None,
        promotion_metric: Optional[str] = None,
        metric_higher_is_better: bool = True,
        auto_evaluate: bool = False,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.payload_name = "checkpoint.pt"
        self.metadata_name = "metadata.json"
        self.index_path = self.root / "index.json"
        self.active_path = active_path.expanduser().resolve() if active_path else None
        self.evaluation_tool = evaluation_tool
        self.evaluation_suite = evaluation_suite
        self.promotion_metric = promotion_metric
        self.metric_higher_is_better = metric_higher_is_better
        self.auto_evaluate = auto_evaluate
        self._mcp_session: Optional[MCPToolSession] = None
        self._index: MutableMapping[str, object] = {"records": []}
        self._load_index()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if self.index_path.exists():
            try:
                with self.index_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                data = {}
            if isinstance(data, Mapping):
                records = data.get("records", [])
                self._index["records"] = [str(r) for r in records if isinstance(r, str)]
                active = data.get("active")
                if isinstance(active, str):
                    self._index["active"] = active

    def _write_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        payload = {
            "records": list(self._index.get("records", [])),
            "active": self._index.get("active"),
        }
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp.replace(self.index_path)

    # ------------------------------------------------------------------
    # MCP integration
    # ------------------------------------------------------------------
    def attach_mcp_session(self, session: MCPToolSession) -> None:
        self._mcp_session = session

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------
    def _record_dir(self, identifier: str) -> Path:
        return self.root / identifier

    def _record_paths(self, identifier: str) -> Optional[tuple[Path, Path]]:
        directory = self._record_dir(identifier)
        payload = directory / self.payload_name
        metadata_path = directory / self.metadata_name
        if not payload.exists() or not metadata_path.exists():
            return None
        return payload, metadata_path

    def get(self, identifier: str) -> Optional[CheckpointRecord]:
        paths = self._record_paths(identifier)
        if paths is None:
            return None
        payload, metadata_path = paths
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            metadata = {}
        return CheckpointRecord(identifier=identifier, payload_path=payload, metadata_path=metadata_path, metadata=metadata)

    def list_records(self) -> Iterable[CheckpointRecord]:
        identifiers = list(self._index.get("records", []))
        for identifier in identifiers:
            record = self.get(identifier)
            if record is not None:
                yield record

    def find_by_payload(self, path: Path) -> Optional[CheckpointRecord]:
        target = path.expanduser().resolve()
        for record in self.list_records():
            try:
                if record.payload_path.exists() and record.payload_path.resolve().samefile(target):
                    return record
            except FileNotFoundError:
                continue
        return None

    def active_record(self) -> Optional[CheckpointRecord]:
        active_id = self._index.get("active")
        if isinstance(active_id, str):
            record = self.get(active_id)
            if record is not None:
                return record
        if self.active_path and self.active_path.exists():
            # Attempt best-effort mapping by checksum of payload path
            for record in self.list_records():
                if record.payload_path.exists() and record.payload_path.samefile(self.active_path):
                    self._index["active"] = record.identifier
                    self._write_index()
                    return record
        return None

    # ------------------------------------------------------------------
    # Creation & metadata management
    # ------------------------------------------------------------------
    def create_checkpoint(
        self,
        payload: Mapping[str, object],
        *,
        step: int,
        reason: str,
        tags: Optional[Sequence[str]] = None,
        extra_metadata: Optional[Mapping[str, object]] = None,
        external_path: Optional[Path] = None,
        auto_evaluate: Optional[bool] = None,
    ) -> CheckpointRecord:
        identifier = self._generate_identifier(step)
        tmp_dir = self.root / f".pending-{identifier}-{uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        payload_path = tmp_dir / self.payload_name
        torch.save(dict(payload), payload_path)
        metadata = self._build_metadata(
            identifier=identifier,
            step=step,
            reason=reason,
            tags=tags,
            extra_metadata=extra_metadata,
        )
        metadata_path = tmp_dir / self.metadata_name
        self._write_metadata(metadata_path, metadata)
        final_dir = self._record_dir(identifier)
        tmp_dir.replace(final_dir)
        record = CheckpointRecord(
            identifier=identifier,
            payload_path=final_dir / self.payload_name,
            metadata_path=final_dir / self.metadata_name,
            metadata=metadata,
        )
        records = list(self._index.get("records", []))
        records.append(identifier)
        self._index["records"] = records
        self._write_index()
        if external_path is not None:
            self._copy_payload(record.payload_path, external_path)
        should_auto = self.auto_evaluate if auto_evaluate is None else auto_evaluate
        if should_auto:
            self.auto_evaluate_record(record)
        return record

    def _write_metadata(self, path: Path, metadata: Mapping[str, object]) -> None:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        tmp.replace(path)

    def _generate_identifier(self, step: int) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"step{step:08d}-{timestamp}"

    def _build_metadata(
        self,
        *,
        identifier: str,
        step: int,
        reason: str,
        tags: Optional[Sequence[str]] = None,
        extra_metadata: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "id": identifier,
            "created_at": _now(),
            "step": int(step),
            "reason": reason,
            "status": "candidate",
            "tags": list(tags or []),
        }
        if extra_metadata:
            metadata["extra"] = dict(extra_metadata)
        return metadata

    def _copy_payload(self, source: Path, target: Path) -> None:
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(".tmp")
        shutil.copy2(source, tmp_target)
        tmp_target.replace(target)

    # ------------------------------------------------------------------
    # Promotion & rollback
    # ------------------------------------------------------------------
    def promote(
        self,
        identifier: str,
        *,
        evaluation_suite: Optional[str] = None,
        metrics: Optional[Mapping[str, object]] = None,
        verdict: Optional[str] = None,
    ) -> Optional[CheckpointRecord]:
        record = self.get(identifier)
        if record is None:
            return None
        metadata = dict(record.metadata)
        now = _now()
        previous = self.active_record()
        if previous and previous.identifier != identifier:
            prev_meta = dict(previous.metadata)
            prev_meta["status"] = "superseded"
            prev_meta["superseded_at"] = now
            prev_meta["superseded_by"] = identifier
            self._write_metadata(previous.metadata_path, prev_meta)
        metadata["status"] = "promoted"
        metadata["promoted_at"] = now
        if previous and previous.identifier != identifier:
            metadata["previous_active"] = previous.identifier
        if evaluation_suite:
            metadata.setdefault("evaluation", {})
            if isinstance(metadata["evaluation"], Mapping):
                evaluation_payload = dict(metadata["evaluation"])
            else:
                evaluation_payload = {}
            evaluation_payload["suite"] = evaluation_suite
            if metrics is not None:
                evaluation_payload["metrics"] = dict(metrics)
            if verdict:
                evaluation_payload["verdict"] = verdict
            evaluation_payload["evaluated_at"] = now
            metadata["evaluation"] = evaluation_payload
        elif verdict:
            metadata["verdict"] = verdict
        self._write_metadata(record.metadata_path, metadata)
        self._index["active"] = identifier
        self._write_index()
        record = CheckpointRecord(
            identifier=identifier,
            payload_path=record.payload_path,
            metadata_path=record.metadata_path,
            metadata=metadata,
        )
        if self.active_path is not None:
            self._copy_payload(record.payload_path, self.active_path)
        return record

    def reject(
        self,
        identifier: str,
        *,
        evaluation_suite: Optional[str] = None,
        metrics: Optional[Mapping[str, object]] = None,
        verdict: Optional[str] = None,
    ) -> Optional[CheckpointRecord]:
        record = self.get(identifier)
        if record is None:
            return None
        metadata = dict(record.metadata)
        metadata["status"] = "rejected"
        metadata["rejected_at"] = _now()
        if evaluation_suite:
            metadata.setdefault("evaluation", {})
            evaluation_data = dict(metadata["evaluation"]) if isinstance(metadata["evaluation"], Mapping) else {}
            evaluation_data["suite"] = evaluation_suite
            if metrics is not None:
                evaluation_data["metrics"] = dict(metrics)
            if verdict:
                evaluation_data["verdict"] = verdict
            evaluation_data["evaluated_at"] = metadata["rejected_at"]
            metadata["evaluation"] = evaluation_data
        elif verdict:
            metadata["verdict"] = verdict
        self._write_metadata(record.metadata_path, metadata)
        return CheckpointRecord(
            identifier=record.identifier,
            payload_path=record.payload_path,
            metadata_path=record.metadata_path,
            metadata=metadata,
        )

    def revert_to(self, identifier: str, *, reason: str) -> Optional[CheckpointRecord]:
        target = self.get(identifier)
        if target is None:
            return None
        current = self.active_record()
        now = _now()
        if current and current.identifier != identifier:
            current_meta = dict(current.metadata)
            current_meta["status"] = "rolled_back"
            current_meta["rolled_back_at"] = now
            current_meta["rollback_reason"] = reason
            current_meta["rolled_back_to"] = identifier
            self._write_metadata(current.metadata_path, current_meta)
        metadata = dict(target.metadata)
        metadata["status"] = "promoted"
        metadata.setdefault("reinstated_at", now)
        metadata["reinstated_reason"] = reason
        self._write_metadata(target.metadata_path, metadata)
        self._index["active"] = identifier
        self._write_index()
        if self.active_path is not None:
            self._copy_payload(target.payload_path, self.active_path)
        return CheckpointRecord(
            identifier=identifier,
            payload_path=target.payload_path,
            metadata_path=target.metadata_path,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def auto_evaluate_record(
        self,
        record: CheckpointRecord,
        *,
        override_suite: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if not force and not self.auto_evaluate:
            return
        tool = self.evaluation_tool
        if not tool or self._mcp_session is None:
            return
        suite = override_suite or self.evaluation_suite
        request = {
            "checkpoint_path": str(record.payload_path),
            "metadata_path": str(record.metadata_path),
            "identifier": record.identifier,
            "step": record.metadata.get("step"),
        }
        if suite:
            request["suite"] = suite
        result = self._call_mcp(tool, request)
        if not isinstance(result, Mapping):
            return
        metrics = result.get("metrics")
        metrics_dict = dict(metrics) if isinstance(metrics, Mapping) else None
        verdict = result.get("verdict") if isinstance(result.get("verdict"), str) else None
        decision = result.get("promote") if isinstance(result.get("promote"), bool) else None
        regress = bool(result.get("regressed"))
        target_id = result.get("revert_to")
        if isinstance(target_id, str):
            self.revert_to(target_id, reason=verdict or "auto-revert")
            return
        suite_name = suite or self.evaluation_suite
        if decision is None and metrics_dict is not None and self.promotion_metric:
            metric_value = metrics_dict.get(self.promotion_metric)
            if metric_value is not None:
                best_metric = self._best_promoted_metric(self.promotion_metric)
                if best_metric is None:
                    decision = True
                else:
                    if self.metric_higher_is_better:
                        decision = metric_value >= best_metric
                    else:
                        decision = metric_value <= best_metric
        if regress and self._index.get("active") == record.identifier:
            previous = record.metadata.get("previous_active")
            if isinstance(previous, str):
                self.revert_to(previous, reason=verdict or "metric regression")
            return
        if decision:
            self.promote(
                record.identifier,
                evaluation_suite=suite_name,
                metrics=metrics_dict,
                verdict=verdict or "auto-promote",
            )
        else:
            self.reject(
                record.identifier,
                evaluation_suite=suite_name,
                metrics=metrics_dict,
                verdict=verdict or "auto-reject",
            )

    def _call_mcp(self, tool_name: str, request: Mapping[str, object]) -> object:
        session = self._mcp_session
        if session is None:
            return None
        call = getattr(session, "call", None)
        if callable(call):
            return call(tool_name, request)
        return session.invoke(tool_name, request)

    def _best_promoted_metric(self, metric_name: str) -> Optional[float]:
        best: Optional[float] = None
        for record in self.list_records():
            status = record.metadata.get("status")
            if status != "promoted":
                continue
            evaluation = record.metadata.get("evaluation")
            if isinstance(evaluation, Mapping):
                metrics = evaluation.get("metrics")
                if isinstance(metrics, Mapping) and metric_name in metrics:
                    try:
                        value = float(metrics[metric_name])
                    except (TypeError, ValueError):
                        continue
                    if best is None:
                        best = value
                    else:
                        if self.metric_higher_is_better:
                            best = max(best, value)
                        else:
                            best = min(best, value)
        return best

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def resolve_load_path(self) -> Optional[Path]:
        active = self.active_record()
        if active is not None:
            return active.payload_path
        if self.active_path and self.active_path.exists():
            return self.active_path
        # Fall back to newest candidate
        identifiers = list(self._index.get("records", []))
        while identifiers:
            identifier = identifiers[-1]
            record = self.get(identifier)
            if record and record.payload_path.exists():
                return record.payload_path
            identifiers.pop()
        return None


__all__ = ["CheckpointManager", "CheckpointRecord"]

