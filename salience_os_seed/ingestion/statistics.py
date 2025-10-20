"""Token statistics utilities for emergent language modeling."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple


class TokenStatistics:
    """Maintains frequency counts for characters and n-grams."""

    def __init__(self) -> None:
        self.char_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[Tuple[str, str]] = Counter()
        self.trigram_counts: Counter[Tuple[str, str, str]] = Counter()
        self.total_chars: int = 0

    def update(self, text: str) -> None:
        tokens = list(text)
        self.char_counts.update(tokens)
        self.total_chars += len(tokens)
        for i in range(len(tokens) - 1):
            self.bigram_counts[(tokens[i], tokens[i + 1])] += 1
        for i in range(len(tokens) - 2):
            self.trigram_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1

    def merge_score(self, pair: Tuple[str, str]) -> float:
        """Estimate information gain for merging a bigram."""

        count = self.bigram_counts.get(pair, 0)
        if count == 0 or self.total_chars == 0:
            return 0.0
        p_xy = count / self.total_chars
        p_x = self.char_counts[pair[0]] / self.total_chars
        p_y = self.char_counts[pair[1]] / self.total_chars
        return max(0.0, p_xy - p_x * p_y)

    def top_merges(self, top_k: int = 32, min_frequency: int = 4) -> Iterable[Tuple[Tuple[str, str], int, float]]:
        candidates = []
        for pair, freq in self.bigram_counts.most_common():
            if freq < min_frequency:
                continue
            gain = self.merge_score(pair)
            if gain <= 0.0:
                continue
            candidates.append((pair, freq, gain))
            if len(candidates) >= top_k:
                break
        return candidates

    def reset(self) -> None:
        self.char_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.total_chars = 0
