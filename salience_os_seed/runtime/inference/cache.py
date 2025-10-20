"""Hint cache for inference-time MCP guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, MutableMapping, Sequence

HINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HintEntry:
    """Cache entry storing MCP hint payloads."""

    version: int
    hints: tuple[str, ...]


class HintCache:
    """Store MCP-provided hints and merge them into prompts."""

    def __init__(
        self,
        storage: MutableMapping[str, HintEntry] | None = None,
        *,
        required_version: int = HINT_SCHEMA_VERSION,
    ) -> None:
        self._storage: MutableMapping[str, HintEntry] = storage or {}
        self._required_version = required_version

    @property
    def required_version(self) -> int:
        """The hint version compatible with the current runtime."""

        return self._required_version

    def store_hint(
        self,
        key: str,
        hints: Iterable[object],
        *,
        version: int | None = None,
    ) -> None:
        """Persist hints for the provided key, pruning stale versions."""

        normalized = tuple(str(hint).strip() for hint in hints if str(hint).strip())
        if not normalized:
            self._storage.pop(key, None)
            return

        entry_version = version if version is not None else self._required_version
        if entry_version != self._required_version:
            # Drop guidance that targets a different schema than we can consume.
            self._storage.pop(key, None)
            return

        self._storage[key] = HintEntry(version=entry_version, hints=normalized)

    def get_hint(self, key: str) -> HintEntry | None:
        """Return the cached hints for ``key`` if they match the active version."""

        entry = self._storage.get(key)
        if entry is None:
            return None
        if entry.version != self._required_version:
            # Automatically purge stale guidance when the schema changes.
            self._storage.pop(key, None)
            return None
        return entry

    def merge_prompt(
        self, prompt_segments: Sequence[str] | str, key: str
    ) -> Sequence[str] | str:
        """Merge cached hints into the provided prompt representation."""

        entry = self.get_hint(key)
        if isinstance(prompt_segments, str):
            base_text = prompt_segments
            if entry is None:
                return base_text
            hints_text = "\n".join(entry.hints)
            if not hints_text:
                return base_text
            if not base_text:
                return hints_text
            return f"{base_text}\n\n{hints_text}"

        merged = list(prompt_segments)
        if entry is not None:
            merged.extend(entry.hints)
        return merged

    def clear(self) -> None:
        """Remove all cached hints."""

        self._storage.clear()


__all__ = ["HINT_SCHEMA_VERSION", "HintCache", "HintEntry"]
