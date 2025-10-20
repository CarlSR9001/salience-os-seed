"""Storage and indexing helpers for the RLM scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from ..core.reflection import WorkspaceViewer


@dataclass
class ScanHit:
    path: str
    snippet: str
    score: float = 0.0


class RLMStore:
    """Thin abstraction over the workspace and scratch area."""

    def __init__(
        self,
        workspace: WorkspaceViewer,
        scratch_root: Optional[str] = None,
    ) -> None:
        self.workspace = workspace
        self.scratch_root = Path(scratch_root).expanduser().resolve() if scratch_root else None
        if self.scratch_root:
            self.scratch_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Scope management
    # ------------------------------------------------------------------
    def root_scope(self, limit: int = 16) -> Sequence[str]:
        entries = self.workspace.list_entries(".")[:limit]
        return [str(entry.path) for entry in entries]

    def describe_scope(self, scope: Sequence[str], limit: int = 8) -> Sequence[str]:
        descriptions: List[str] = []
        for identifier in scope[:limit]:
            try:
                entries = self.workspace.list_entries(identifier)
            except Exception:
                entries = []
            summary = ", ".join(f"{entry.path} ({entry.size}B{' dir' if entry.is_dir else ''})" for entry in entries[:6])
            if summary:
                descriptions.append(f"{identifier}: {summary}")
        return descriptions

    # ------------------------------------------------------------------
    # Tool primitives
    # ------------------------------------------------------------------
    def read(self, identifier: str, start: Optional[int] = None, end: Optional[int] = None) -> str:
        text = self.workspace.read_text(identifier)
        if start is None and end is None:
            return text
        start = max(0, int(start or 0))
        end = int(end) if end is not None else len(text)
        end = max(start, min(end, len(text)))
        return text[start:end]

    def scan(self, query: str, top_k: int = 5) -> Sequence[Mapping[str, object]]:
        hits = self.workspace.search(query, max_results=top_k)
        scored: List[Mapping[str, object]] = []
        for idx, hit in enumerate(hits):
            scored.append({
                "path": str(hit.get("path", "")),
                "snippet": hit.get("snippet", ""),
                "score": float(top_k - idx) / max(1, top_k),
            })
        return scored

    def summarize(self, identifier: str, target_len: int = 400) -> str:
        text = self.workspace.read_text(identifier)
        target_len = max(120, min(target_len, 1200))
        if len(text) <= target_len:
            return text
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        summary_parts: List[str] = []
        for paragraph in paragraphs:
            if len(" ".join(summary_parts)) >= target_len:
                break
            summary_parts.append(paragraph[: target_len - len(" ".join(summary_parts))])
        if not summary_parts:
            summary_parts = [text[:target_len]]
        return "\n".join(summary_parts)

    def write(self, path: Optional[str], data: object) -> Mapping[str, object]:
        if self.scratch_root is None:
            raise RuntimeError("Scratch root not configured for write tool")
        relative = Path(path or "notes/auto_note.txt")
        safe_path = (self.scratch_root / relative).resolve()
        if not str(safe_path).startswith(str(self.scratch_root)):
            raise PermissionError("Attempted write outside scratch root")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else str(data)
        safe_path.write_text(text, encoding="utf-8")
        return {"ok": True, "id": str(safe_path.relative_to(self.scratch_root))}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def clamp_scope(self, identifiers: Optional[Iterable[str]]) -> Sequence[str]:
        if not identifiers:
            return []
        cleaned: List[str] = []
        for identifier in identifiers:
            if not identifier:
                continue
            cleaned.append(str(identifier))
        return cleaned
