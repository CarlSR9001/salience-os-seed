"""Salience-based filtering utilities for corpus ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from ..core.sensors import SensorBank


@dataclass
class IngestionThresholds:
    """Thresholds applied to salience readings."""

    enabled: bool = False
    min_uncertainty: float = 0.0
    min_novelty: float = 0.0
    max_drag: float = 1.0


class SalienceFilter:
    """Evaluate raw text states using the salience sensor bank."""

    def __init__(self, thresholds: IngestionThresholds, sensor_bank: Optional[SensorBank] = None) -> None:
        self.thresholds = thresholds
        self.sensor_bank = sensor_bank or SensorBank.default_bank()

    def evaluate(
        self,
        state: Mapping[str, object],
        memory_snapshot: Mapping[str, object],
        meta_snapshot: Mapping[str, float],
    ) -> Tuple[bool, Mapping[str, float]]:
        salience_vector = self.sensor_bank.tick(state, memory_snapshot, meta_snapshot)
        readings = salience_vector.as_mapping()
        accept = self._accept(readings)
        return accept, readings

    def _accept(self, readings: Mapping[str, float]) -> bool:
        t = self.thresholds
        if not t.enabled:
            return True
        uncertainty = float(readings.get("uncertainty", 0.0))
        novelty = float(readings.get("novelty", 0.0))
        drag = float(readings.get("drag", 0.0))
        if uncertainty < t.min_uncertainty:
            return False
        if novelty < t.min_novelty:
            return False
        if drag > t.max_drag:
            return False
        return True
