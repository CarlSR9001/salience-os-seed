"""AIM alignment sensor measuring goal adherence.

The sensor consumes embedding vectors for the current prompt/task state and the
explicit goal description. The runtime is responsible for providing embeddings
(e.g., via a lightweight encoder or cached vectors). We compute a cosine
similarity and optionally smooth the output to reduce jitter.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Mapping, MutableMapping, Sequence

import numpy as np

from .base import MedianMADNormalizer, Sensor


class AlignmentSensor(Sensor):
    """Estimate AIM (alignment with the stated goal).

    The runtime should expose `state["embeddings"]["prompt"]` and
    `memory["goal"]["embedding"]` as 1D float arrays. The cosine similarity ranges
    between -1 and 1; we rescale to `[0, 1]` before normalisation.
    """

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        smoothing_window: int = 8,
    ) -> None:
        super().__init__(name="alignment", domain="alignment", normaliser=normaliser)
        if smoothing_window < 4:
            raise ValueError("smoothing_window must be >= 4 for alignment sensor")
        self._window = smoothing_window
        self._buffer: Deque[float] = deque(maxlen=smoothing_window)

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        prompt_vec = _extract_vector(state, ("embeddings", "prompt"))
        goal_vec = _extract_vector(memory, ("goal", "embedding"))
        if prompt_vec is None or goal_vec is None:
            return 0.0
        similarity = _cosine_similarity(prompt_vec, goal_vec)
        # rescale from [-1, 1] to [0, 1]
        scaled = 0.5 * (similarity + 1.0)
        self._buffer.append(scaled)
        if len(self._buffer) < 4:
            return scaled
        return float(np.mean(self._buffer))

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        buffer_min = float(min(self._buffer)) if self._buffer else raw_value
        buffer_max = float(max(self._buffer)) if self._buffer else raw_value
        return {
            "raw_alignment": raw_value,
            "window_min": buffer_min,
            "window_max": buffer_max,
            "normalised": norm_value,
        }


def _extract_vector(source: Mapping[str, object], path: Sequence[str]) -> np.ndarray | None:
    cursor: object = source
    for key in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(key)
    if cursor is None:
        return None
    arr = np.asarray(cursor, dtype=np.float64)
    if arr.ndim != 1:
        return None
    return arr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)
