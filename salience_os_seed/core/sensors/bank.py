"""Sensor bank orchestrating salience vector construction.

The `SensorBank` owns a registry of concrete sensors, handles tick ordering, and
emits the canonical salience vector consumed by the controller. Each tick it:
1. Invokes registered sensors (respecting priority ordering where required).
2. Concatenates normalised values into a deterministic vector.
3. Returns both the vector and per-sensor `SensorReading` objects for logging.

The bank is also responsible for exposing current robust statistics for tests
and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .base import MedianMADNormalizer, Sensor, SensorReading


@dataclass(frozen=True)
class SalienceVector:
    """Container bundling the salience vector and per-sensor readings."""

    values: np.ndarray
    readings: Sequence[SensorReading]

    def as_mapping(self) -> Mapping[str, float]:
        return {reading.name: reading.normalised for reading in self.readings}


class SensorBank:
    """Manage a collection of sensors and emit the salience vector."""

    def __init__(self, sensors: Iterable[Sensor]) -> None:
        self._sensors: List[Sensor] = list(sensors)
        if not self._sensors:
            raise ValueError("SensorBank requires at least one sensor")
        self._ordering = [sensor.name for sensor in self._sensors]

    @classmethod
    def default_bank(cls) -> "SensorBank":
        """Convenience constructor with canonical sensors and shared normaliser."""

        normaliser = MedianMADNormalizer()
        normaliser.register_baseline("uncertainty", [0.6, 1.0, 1.4, 1.8])
        normaliser.register_baseline("novelty", [0.1, 0.3, 0.6, 0.9])
        normaliser.register_baseline("alignment", [0.2, 0.5, 0.8])
        normaliser.register_baseline("progress", [-0.5, 0.0, 0.5, 1.0])
        normaliser.register_baseline("cost", [20.0, 40.0, 80.0, 120.0])
        normaliser.register_baseline("drag", [0.0, 0.3, 0.6, 1.0])
        normaliser.register_baseline("truth", [0.2, 0.4, 0.6, 0.85])
        normaliser.register_baseline("coherence", [0.3, 0.6, 0.8, 1.0])
        from .uncertainty import UncertaintySensor
        from .novelty import NoveltySensor
        from .alignment import AlignmentSensor
        from .progress import ProgressSensor
        from .cost import CostSensor
        from .drag import DragSensor
        from .coherence import CoherenceSensor
        from .truth import TruthSensor

        sensors: List[Sensor] = [
            UncertaintySensor(normaliser=normaliser),
            NoveltySensor(normaliser=normaliser),
            AlignmentSensor(normaliser=normaliser),
            ProgressSensor(normaliser=normaliser),
            CostSensor(normaliser=normaliser),
            DragSensor(normaliser=normaliser),
            TruthSensor(normaliser=normaliser),
            CoherenceSensor(normaliser=normaliser),
        ]
        return cls(sensors)

    def tick(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> SalienceVector:
        readings: List[SensorReading] = []
        values: List[float] = []
        for sensor in self._sensors:
            reading = sensor.tick(state=state, memory=memory, meta=meta)
            readings.append(reading)
            values.append(reading.normalised)
        vector = np.asarray(values, dtype=np.float32)
        return SalienceVector(values=vector, readings=tuple(readings))

    @property
    def ordering(self) -> Tuple[str, ...]:
        return tuple(self._ordering)

    def snapshot_statistics(self) -> Mapping[str, Mapping[str, float]]:
        """Aggregate statistics from each sensor's normaliser."""

        stats = {}
        for sensor in self._sensors:
            normaliser = getattr(sensor, "_normaliser", None)
            if normaliser is None:
                continue
            stats[sensor.name] = normaliser.snapshot_statistics().get(sensor.name, {})
        return stats
