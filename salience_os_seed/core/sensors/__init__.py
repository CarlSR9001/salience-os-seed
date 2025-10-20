"""Sensor modules for SalienceOS Seed v0.1.

The package exposes concrete sensor implementations alongside the shared registry
used by the salience bank to produce the canonical salience vector. Each sensor
tracks a different aspect of the system state (uncertainty, novelty, alignment,
progress, cost, drag) and reports a robustly normalised scalar.
"""

from .base import Sensor, SensorReading, MedianMADNormalizer
from .uncertainty import UncertaintySensor
from .novelty import NoveltySensor
from .alignment import AlignmentSensor
from .progress import ProgressSensor
from .cost import CostSensor
from .drag import DragSensor
from .bank import SensorBank

__all__ = [
    "Sensor",
    "SensorReading",
    "MedianMADNormalizer",
    "UncertaintySensor",
    "NoveltySensor",
    "AlignmentSensor",
    "ProgressSensor",
    "CostSensor",
    "DragSensor",
    "SensorBank",
]
