"""Vocabulary construction for emergent symbol discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..ingestion.statistics import TokenStatistics


@dataclass
class Vocabulary:
    """Byte-Pair style vocabulary updated via token statistics."""

    tokens: List[str] = field(default_factory=list)
    merges: List[Tuple[str, str]] = field(default_factory=list)
    token_to_id: Dict[str, int] = field(default_factory=dict)
    merge_map: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    locked: bool = False

    def __post_init__(self) -> None:
        if not self.tokens:
            # initialize with ASCII + whitespace basics
            base_chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
            self.tokens.extend(base_chars)
        if not self.merge_map and self.merges:
            for pair in self.merges:
                self.merge_map["".join(pair)] = pair
        self._refresh_index()

    def _refresh_index(self) -> None:
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}

    def add_merge(self, pair: Tuple[str, str]) -> None:
        merged = "".join(pair)
        if merged in self.token_to_id:
            return
        self.tokens.append(merged)
        self.merges.append(pair)
        self.merge_map[merged] = pair
        self._refresh_index()

    def build_from_statistics(self, stats: TokenStatistics, merges: int = 64) -> None:
        for pair, _, _ in stats.top_merges(top_k=merges):
            self.add_merge(pair)

    def encode(self, text: str) -> List[str]:
        symbols = list(text)
        idx = 0
        while idx < len(symbols) - 1:
            pair = (symbols[idx], symbols[idx + 1])
            if pair in self.merges:
                merged = "".join(pair)
                symbols[idx : idx + 2] = [merged]
                continue
            idx += 1
        return symbols

    def encode_ids(self, text: str) -> List[int]:
        ids: List[int] = []
        for token in self.encode(text):
            token_id = self.token_to_id.get(token)
            if token_id is None:
                token_id = self._add_token(token)
            ids.append(token_id)
        return ids

    def decode(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def decode_ids(self, ids: Sequence[int]) -> str:
        tokens = [self.tokens[idx] for idx in ids if 0 <= idx < len(self.tokens)]
        return self.decode(tokens)

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.locked:
            raise RuntimeError(f"Vocabulary is locked; cannot add token '{token}'")
        self.tokens.append(token)
        self._refresh_index()
        return self.token_to_id[token]

    def size(self) -> int:
        return len(self.tokens)

    def drop_ids(self, indices: Sequence[int]) -> List[str]:
        to_remove = sorted(set(idx for idx in indices if 0 <= idx < len(self.tokens)), reverse=True)
        if not to_remove:
            return []

        removed_tokens: List[str] = []
        for idx in to_remove:
            removed_tokens.append(self.tokens[idx])
            del self.tokens[idx]

        remaining = set(self.tokens)
        self.merges = [pair for pair in self.merges if pair[0] in remaining and pair[1] in remaining]
        self._refresh_index()
        self.merge_map = {
            token: pair
            for token, pair in self.merge_map.items()
            if token in self.token_to_id and pair[0] in self.token_to_id and pair[1] in self.token_to_id
        }
        for token in removed_tokens:
            self.merge_map.pop(token, None)
        return list(reversed(removed_tokens))

    def parent_tokens(self, token: str) -> Optional[Tuple[str, str]]:
        return self.merge_map.get(token)

    def parent_ids(self, token_id: int) -> List[int]:
        if token_id < 0 or token_id >= len(self.tokens):
            return []
        parents = self.parent_tokens(self.tokens[token_id])
        if not parents:
            return []
        ids: List[int] = []
        for symbol in parents:
            idx = self.token_to_id.get(symbol)
            if idx is not None:
                ids.append(idx)
        return ids

    def clone(self) -> "Vocabulary":
        clone_vocab = Vocabulary(
            tokens=list(self.tokens),
            merges=list(self.merges),
            merge_map=dict(self.merge_map),
            locked=self.locked,
        )
        return clone_vocab

    def lock(self) -> None:
        self.locked = True

    def unlock(self) -> None:
        self.locked = False
