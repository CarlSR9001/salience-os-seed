"""Common primitives for salience sensors.

This module defines the abstract `Sensor` contract alongside:
- `SensorReading`: immutable container exposing raw and normalised sensor outputs.
- `MedianMADNormalizer`: robust domain-specific normaliser using a rolling window of
  samples to estimate median and median absolute deviation (MAD).

Every concrete sensor inherits from `Sensor` and implements `_measure()` to produce
raw scalars from runtime state. The base class handles timestamping, robust
normalisation, and metadata capture.
"""

from __future__ import annotations

import abc
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np


@dataclass(frozen=True)
class SensorReading:
    """Represents a single sensor emission.

    Attributes
    ----------
    name:
        Human-readable identifier (e.g., "uncertainty").
    raw:
        Raw scalar measurement before normalisation.
    normalised:
        Robustly normalised value (median/MAD) used throughout the system.
    timestamp:
        Epoch seconds when the reading was produced.
    metadata:
        Optional auxiliary context for downstream modules (kept minimal to avoid
        bloating the salience vector while still enabling diagnostics).
    """

    name: str
    raw: float
    normalised: float
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Mapping[str, float] | None = None


class MedianMADNormalizer:
    r"""Robustly normalise scalar streams via median and MAD.

    The normaliser maintains a fixed-length buffer per `domain` (logical key). For
    each new sample it computes:

    \f[
        \hat{x} = \frac{x - \text{median}}{\max(\text{MAD}, \epsilon)}
    \f]

    where MAD is `median(|x - median|)` using the buffer contents. A small
    `epsilon` prevents division by zero when the stream is degenerate.
    """

    def __init__(
        self,
        window: int = 128,
        epsilon: float = 1e-6,
    ) -> None:
        if window < 8:
            raise ValueError("MedianMADNormalizer window must be >= 8 for stability")
        self._window = window
        self._epsilon = epsilon
        self._buffers: MutableMapping[str, Deque[float]] = {}

    def _get_buffer(self, domain: str) -> Deque[float]:
        if domain not in self._buffers:
            self._buffers[domain] = deque(maxlen=self._window)
        return self._buffers[domain]

    def register_baseline(self, domain: str, samples: Iterable[float]) -> None:
        """Seed the normaliser with baseline samples.

        Baselines accelerate convergence when we already have historical ranges
        (e.g., from calibration datasets)."""

        buffer = self._get_buffer(domain)
        for value in samples:
            buffer.append(float(value))

    def normalise(self, domain: str, value: float) -> float:
        buffer = self._get_buffer(domain)
        buffer.append(float(value))
        arr = np.fromiter(buffer, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        denom = mad if mad > self._epsilon else self._epsilon
        return (float(value) - median) / denom

    def snapshot_statistics(self) -> Dict[str, Dict[str, float]]:
        """Return current median and MAD for diagnostics/testing."""

        stats: Dict[str, Dict[str, float]] = {}
        for domain, buffer in self._buffers.items():
            if not buffer:
                continue
            arr = np.fromiter(buffer, dtype=np.float64)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            stats[domain] = {"median": median, "mad": mad}
        return stats


class Sensor(abc.ABC):
    """Abstract salience sensor.

    Concrete subclasses implement `_measure(state, memory, meta)` and optionally
    `_metadata(...)`. The base class wires in normalisation, timestamping, and
    diagnostics. Sensors are stateful only through their normaliser and any
    internal buffers needed for measurement (e.g., rolling entropy trackers).
    """

    def __init__(self, name: str, domain: str, normaliser: MedianMADNormalizer) -> None:
        self._name = name
        self._domain = domain
        self._normaliser = normaliser
        self._last_reading: Optional[SensorReading] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_reading(self) -> Optional[SensorReading]:
        return self._last_reading

    def tick(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> SensorReading:
        raw_value = float(self._measure(state, memory, meta))
        norm_value = float(self._normaliser.normalise(self._domain, raw_value))
        reading = SensorReading(
            name=self._name,
            raw=raw_value,
            normalised=norm_value,
            metadata=self._metadata(state, memory, meta, raw_value, norm_value),
        )
        self._last_reading = reading
        return reading

    @abc.abstractmethod
    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        """Return the raw scalar for the current tick."""

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> Mapping[str, float] | None:
        """Optional hook for attaching diagnostic metadata."""

        return None
