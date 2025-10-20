"""Self-awareness meta-state maintenance.

The meta-state tracks slow-moving cognitive signals such as calibrated
confidence, estimated remaining difficulty, blind-spot fingerprints, and
predicted ROI of deeper computation. A lightweight GRU cell updates the
meta-vector using the current salience readings plus verification outcomes.

The class purposely avoids heavyweight dependencies: updates rely on NumPy and
hand-written GRU equations, keeping the seed deployable in minimalist
environments while remaining faithful to the intended dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Optional

import numpy as np


@dataclass
class MetaStateConfig:
    """Configuration for the meta-state dynamics."""

    vector_dim: int = 8
    salience_dim: int = 6
    history_window: int = 64
    gru_init_scale: float = 0.1
    confidence_index: int = 0
    difficulty_index: int = 1
    blindspot_index: int = 2
    roi_index: int = 3


@dataclass
class MetaState:
    """GRU-backed meta-state vector."""

    config: MetaStateConfig
    vector: np.ndarray = field(init=False)
    gru_weights: Mapping[str, np.ndarray] = field(init=False)
    history: MutableMapping[str, list[float]] = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        self.vector = np.zeros(cfg.vector_dim, dtype=np.float32)
        self.gru_weights = self._initialise_weights(
            input_dim=cfg.salience_dim + 3,  # include outcome + budget signals
            hidden_dim=cfg.vector_dim,
            scale=cfg.gru_init_scale,
        )
        self.history = {
            "confidence": [],
            "difficulty": [],
            "roi": [],
        }

    def update(
        self,
        salience_vector: Mapping[str, float],
        verification_passed: Optional[bool],
        budget_left: float,
        cooldown_active: bool,
    ) -> np.ndarray:
        """Update the meta-state based on current signals."""

        x = self._construct_input(
            salience_vector=salience_vector,
            verification_passed=verification_passed,
            budget_left=budget_left,
            cooldown_active=cooldown_active,
        )
        self.vector = self._gru_step(x, self.vector)
        self._log_history()
        self._apply_constraints()
        return self.vector.copy()

    def snapshot(self) -> Mapping[str, float]:
        """Expose human-readable summary for logging or UI."""

        cfg = self.config
        return {
            "confidence": float(self.vector[cfg.confidence_index]),
            "difficulty": float(self.vector[cfg.difficulty_index]),
            "blind_spot": float(self.vector[cfg.blindspot_index]),
            "roi": float(self.vector[cfg.roi_index]),
        }

    def _construct_input(
        self,
        salience_vector: Mapping[str, float],
        verification_passed: Optional[bool],
        budget_left: float,
        cooldown_active: bool,
    ) -> np.ndarray:
        values = [float(salience_vector.get(key, 0.0)) for key in sorted(salience_vector)]
        outcome = 1.0 if verification_passed is True else -1.0 if verification_passed is False else 0.0
        cooldown = 1.0 if cooldown_active else 0.0
        values.extend([outcome, float(budget_left), cooldown])
        return np.asarray(values, dtype=np.float32)

    def _gru_step(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        w = self.gru_weights
        z = self._sigmoid(x @ w["xz"] + h_prev @ w["hz"] + w["bz"])
        r = self._sigmoid(x @ w["xr"] + h_prev @ w["hr"] + w["br"])
        h_candidate = np.tanh(x @ w["xh"] + (r * h_prev) @ w["hh"] + w["bh"])
        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new.astype(np.float32)

    def _apply_constraints(self) -> None:
        cfg = self.config
        self.vector[cfg.confidence_index] = np.clip(self.vector[cfg.confidence_index], -1.0, 1.0)
        self.vector[cfg.difficulty_index] = np.clip(self.vector[cfg.difficulty_index], 0.0, 2.0)
        self.vector[cfg.roi_index] = np.clip(self.vector[cfg.roi_index], -1.0, 2.0)

    def _log_history(self) -> None:
        cfg = self.config
        window = cfg.history_window
        for key, index in (
            ("confidence", cfg.confidence_index),
            ("difficulty", cfg.difficulty_index),
            ("roi", cfg.roi_index),
        ):
            buffer = self.history[key]
            buffer.append(float(self.vector[index]))
            if len(buffer) > window:
                del buffer[0]

    @staticmethod
    def _initialise_weights(input_dim: int, hidden_dim: int, scale: float) -> Mapping[str, np.ndarray]:
        rng = np.random.default_rng(seed=42)
        return {
            "xz": rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32),
            "hz": rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32),
            "bz": np.zeros(hidden_dim, dtype=np.float32),
            "xr": rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32),
            "hr": rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32),
            "br": np.zeros(hidden_dim, dtype=np.float32),
            "xh": rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32),
            "hh": rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32),
            "bh": np.zeros(hidden_dim, dtype=np.float32),
        }

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
