"""Read-only workspace viewer for reflective access to project assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class WorkspaceEntry:
    path: Path
    is_dir: bool
    size: int


class WorkspaceViewer:
    """Expose a read-only snapshot of a configured workspace directory."""

    def __init__(self, root: str | Path, include_hidden: bool = False, max_bytes: int = 5_000_000) -> None:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"Workspace root '{root_path}' does not exist or is not a directory")
        self._root = root_path
        self._include_hidden = include_hidden
        self._max_bytes = max_bytes

    @property
    def root(self) -> Path:
        return self._root

    def list_entries(self, relative_path: str | Path = ".") -> List[WorkspaceEntry]:
        target = self._resolve(relative_path)
        if not target.is_dir():
            raise NotADirectoryError(f"'{target}' is not a directory")
        entries: List[WorkspaceEntry] = []
        for path in sorted(target.iterdir()):
            if not self._include_hidden and path.name.startswith("."):
                continue
            stat = path.stat()
            entries.append(WorkspaceEntry(path=self._relativize(path), is_dir=path.is_dir(), size=stat.st_size))
        return entries

    def read_text(self, relative_path: str | Path, encoding: str = "utf-8", max_chars: Optional[int] = None) -> str:
        file_path = self._resolve(relative_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"'{file_path}' is not a file")
        size = file_path.stat().st_size
        if size > self._max_bytes:
            raise ValueError(f"File '{file_path}' exceeds max readable size ({self._max_bytes} bytes)")
        content = file_path.read_text(encoding=encoding)
        if max_chars is not None and len(content) > max_chars:
            return content[:max_chars]
        return content

    def search(self, keyword: str, max_results: int = 20) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for path in self._root.rglob("*"):
            if len(results) >= max_results:
                break
            if not path.is_file():
                continue
            if not self._include_hidden and any(part.startswith(".") for part in path.relative_to(self._root).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if keyword.lower() in text.lower():
                rel = self._relativize(path)
                results.append({
                    "path": rel,
                    "snippet": self._extract_snippet(text, keyword),
                })
        return results

    def _resolve(self, relative_path: str | Path) -> Path:
        candidate = (self._root / Path(relative_path)).resolve()
        if not str(candidate).startswith(str(self._root)):
            raise PermissionError("Attempted access outside workspace root")
        return candidate

    def _relativize(self, path: Path) -> Path:
        return path.relative_to(self._root)

    @staticmethod
    def _extract_snippet(text: str, keyword: str, window: int = 80) -> str:
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return text[:window]
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet
