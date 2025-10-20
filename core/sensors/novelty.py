"""Novelty sensor capturing surprisal deltas and n-gram freshness.

The sensor combines two signals:
1. Token-level surprisal delta versus a frozen lightweight baseline or rolling
   average. This highlights moments where the model is seeing unexpected tokens.
2. An n-gram novelty score derived from a Bloom-filter style recent history.

The final raw score is a weighted geometric mean of the scaled surprisal delta
and the n-gram freshness. This keeps the metric sensitive to both semantic and
syntactic novelty while remaining bounded for robust MAD normalisation.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .base import MedianMADNormalizer, Sensor


class NoveltySensor(Sensor):
    """Track NEW (novelty) by mixing surprisal spikes with n-gram freshness."""

    def __init__(
        self,
        normaliser: MedianMADNormalizer,
        baseline_surprisal: float = 3.5,
        rolling_window: int = 64,
        ngram: int = 4,
        freshness_decay: float = 0.9,
    ) -> None:
        super().__init__(name="novelty", domain="novelty", normaliser=normaliser)
        if rolling_window < 16:
            raise ValueError("rolling_window must be >= 16 for stable novelty estimates")
        if ngram < 2:
            raise ValueError("ngram must be >= 2 for novelty tracking")
        self._baseline_surprisal = float(baseline_surprisal)
        self._rolling_window = rolling_window
        self._surprisal_buffer: Deque[float] = deque(maxlen=rolling_window)
        self._ngram = ngram
        self._freshness_decay = freshness_decay
        self._ngram_counter: Counter[str] = Counter()
        self._recent_phrases: Deque[str] = deque(maxlen=rolling_window)

    def _measure(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
    ) -> float:
        surprisal = _extract_surprisal(state)
        surprisal_delta = max(surprisal - self._baseline_surprisal, 0.0)
        self._surprisal_buffer.append(surprisal)

        tokens = _extract_tokens(state)
        freshness = self._update_freshness(tokens)

        delta_scale = math.tanh(surprisal_delta)
        novelty_raw = math.sqrt(delta_scale * freshness)
        return novelty_raw

    def _metadata(
        self,
        state: Mapping[str, object],
        memory: Mapping[str, object],
        meta: Mapping[str, object],
        raw_value: float,
        norm_value: float,
    ) -> MutableMapping[str, float]:
        buffer_mean = float(np.mean(self._surprisal_buffer)) if self._surprisal_buffer else 0.0
        buffer_std = float(np.std(self._surprisal_buffer)) if self._surprisal_buffer else 0.0
        return {
            "raw_novelty": raw_value,
            "surprisal_mean": buffer_mean,
            "surprisal_std": buffer_std,
            "ngram_unique": float(len(self._ngram_counter)),
            "normalised": norm_value,
        }

    def _update_freshness(self, tokens: Sequence[str]) -> float:
        if len(tokens) < self._ngram:
            return 0.0
        phrase = "::".join(tokens[-self._ngram :])
        count = self._ngram_counter[phrase]
        freshness = 1.0 / (1.0 + count)
        self._ngram_counter[phrase] = count + 1
        self._recent_phrases.append(phrase)
        if len(self._recent_phrases) == self._recent_phrases.maxlen:
            oldest = self._recent_phrases[0]
            self._ngram_counter[oldest] *= self._freshness_decay
            if self._ngram_counter[oldest] < 0.1:
                del self._ngram_counter[oldest]
        return freshness


def _extract_surprisal(state: Mapping[str, object]) -> float:
    prediction = state.get("prediction")
    if isinstance(prediction, Mapping):
        if "surprisal" in prediction:
            return float(prediction["surprisal"])
        logprob = prediction.get("token_logprob")
        if logprob is not None:
            return float(-logprob)
        logprobs = prediction.get("token_logprobs")
        if isinstance(logprobs, Iterable):
            last = float(list(logprobs)[-1])
            return float(-last)
    return float(state.get("fallback_surprisal", 0.0))


def _extract_tokens(state: Mapping[str, object]) -> Sequence[str]:
    context = state.get("context")
    if isinstance(context, Mapping):
        tokens = context.get("tokens")
        if isinstance(tokens, Sequence) and tokens:
            return list(tokens)
        text = context.get("text")
        if isinstance(text, str):
            return text.split()
    return []
