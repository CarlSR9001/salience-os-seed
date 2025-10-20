"""Adaptive parameter vault tailored for the seed runtime.

This module mirrors the functionality of `parameter_vault.vault` but is scoped to the
seed runtime. It focuses on recording weight payloads emitted during online
training and choosing which variants to keep active.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..core.spatial import FourDCoordinate, payload_to_coordinate
from ..telemetry import BUS, SpatialEvent


class WeightProvenance(str, Enum):
    CLONE = "clone"
    EXTENSION = "extension"
    AUTONOMOUS = "autonomous"
    INFERRED = "inferred"


@dataclass
class WeightSnapshot:
    weight_id: str
    payload_hash: str
    payload: Dict[str, Any]
    provenance: WeightProvenance
    created_at: float
    salience_scores: Dict[str, float]
    notes: Dict[str, Any] = field(default_factory=dict)
    coordinate: Optional[FourDCoordinate] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_id": self.weight_id,
            "payload_hash": self.payload_hash,
            "payload": self.payload,
            "provenance": self.provenance.value,
            "created_at": self.created_at,
            "salience_scores": self.salience_scores,
            "notes": self.notes,
            "coordinate": self.coordinate.to_dict() if self.coordinate else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightSnapshot":
        coord_payload = data.get("coordinate")
        coord = None
        if isinstance(coord_payload, dict):
            coord = FourDCoordinate(
                float(coord_payload.get("x", 0.0)),
                float(coord_payload.get("y", 0.0)),
                float(coord_payload.get("z", 0.0)),
                float(coord_payload.get("w", 0.0)),
            )
        return cls(
            weight_id=data["weight_id"],
            payload_hash=data.get("payload_hash", ""),
            payload=data.get("payload", {}),
            provenance=WeightProvenance(data.get("provenance", WeightProvenance.INFERRED.value)),
            created_at=float(data.get("created_at", 0.0)),
            salience_scores=dict(data.get("salience_scores", {})),
            notes=dict(data.get("notes", {})),
            coordinate=coord,
        )


@dataclass
class VaultStats:
    total: int
    by_provenance: Dict[str, int]
    avg_salience: Dict[str, float]


class AdaptiveVault:
    def __init__(self, *, capacity: int = 512) -> None:
        self.capacity = capacity
        self._weights: Dict[str, WeightSnapshot] = {}

    def register(
        self,
        payload: Dict[str, Any],
        *,
        salience_scores: Dict[str, float],
        provenance: WeightProvenance,
        notes: Optional[Dict[str, Any]] = None,
    ) -> WeightSnapshot:
        payload_hash = self._hash_payload(payload)
        weight_id = f"w_{payload_hash[:10]}"
        snapshot = WeightSnapshot(
            weight_id=weight_id,
            payload_hash=payload_hash,
            payload=payload,
            provenance=provenance,
            created_at=time.time(),
            salience_scores=salience_scores.copy(),
            notes=notes.copy() if notes else {},
            coordinate=payload_to_coordinate(payload),
        )
        self._ensure_capacity()
        self._weights[weight_id] = snapshot
        if snapshot.coordinate:
            BUS.publish(
                SpatialEvent(
                    payload={
                        "space": "parameters",
                        "weight_id": weight_id,
                        "summary": f"salience={salience_scores.get('payoff', 0.0):.3f}",
                        "coordinate": snapshot.coordinate.to_dict(),
                    }
                )
            )
        return snapshot

    def promote(self, weight_id: str, *, note: str) -> None:
        snapshot = self._weights.get(weight_id)
        if not snapshot:
            return
        history = snapshot.notes.setdefault("promotion_history", [])
        history.append({"timestamp": time.time(), "note": note})

    def drop(self, weight_id: str) -> bool:
        return self._weights.pop(weight_id, None) is not None

    def best_candidate(self, *, key: str = "payoff") -> Optional[WeightSnapshot]:
        if not self._weights:
            return None
        return max(
            self._weights.values(),
            key=lambda snap: snap.salience_scores.get(key, 0.0),
        )

    def stats(self) -> VaultStats:
        by_provenance: Dict[str, int] = {prov.value: 0 for prov in WeightProvenance}
        totals: Dict[str, float] = {}
        for snapshot in self._weights.values():
            by_provenance[snapshot.provenance.value] += 1
            for key, value in snapshot.salience_scores.items():
                totals.setdefault(key, 0.0)
                totals[key] += value
        avg_salience = {
            key: value / len(self._weights) if self._weights else 0.0
            for key, value in totals.items()
        }
        return VaultStats(total=len(self._weights), by_provenance=by_provenance, avg_salience=avg_salience)

    def list_all(self) -> Dict[str, WeightSnapshot]:
        return self._weights.copy()

    def serialize(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "weights": [snapshot.to_dict() for snapshot in self._weights.values()],
        }

    def restore(self, payload: Dict[str, Any]) -> None:
        self.capacity = int(payload.get("capacity", self.capacity))
        self._weights.clear()
        for entry in payload.get("weights", []):
            snapshot = WeightSnapshot.from_dict(entry)
            self._weights[snapshot.weight_id] = snapshot

    def _ensure_capacity(self) -> None:
        if len(self._weights) < self.capacity:
            return
        oldest_id = min(self._weights.items(), key=lambda item: item[1].created_at)[0]
        self._weights.pop(oldest_id, None)

    @staticmethod
    def _hash_payload(payload: Dict[str, Any]) -> str:
        text = repr(sorted(payload.items()))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
