"""File ingestion pipeline for incremental training."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True)
class WorkEstimate:
    total_files: int
    total_chunks: int


class CorpusReader:
    """Iterate over eligible text files and emit chunk metadata."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".jsonl"}

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Corpus root '{self.root}' does not exist")
        self._files = self._collect_files()
        self._sizes = {path: max(path.stat().st_size, 0) for path in self._files}

    def _collect_files(self) -> Tuple[Path, ...]:
        files = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(path)
        return tuple(files)

    def files(self) -> Iterable[Path]:
        return self._files

    def estimate_work(self, chunk_size: int = 2048, chunk_overlap: int = 0) -> WorkEstimate:
        total_chunks = 0
        for path in self._files:
            file_size = self._sizes[path]
            stride = max(1, chunk_size - chunk_overlap)
            total_chunks += max(1, math.ceil(max(file_size - chunk_overlap, 0) / stride)) if file_size else 1
        return WorkEstimate(total_files=len(self._files), total_chunks=total_chunks)

    def stream(
        self,
        chunk_size: int = 2048,
        *,
        chunk_overlap: int = 0,
        batch_size: int = 1,
    ) -> Iterator[List[Tuple[dict, str]]]:
        total_files = len(self._files)
        stride = max(1, chunk_size - chunk_overlap)
        batch_cap = max(1, batch_size)
        for file_index, path in enumerate(self._files, start=1):
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            if not text:
                continue
            windows: List[str] = []
            if len(text) <= chunk_size:
                windows = [text]
            else:
                starts = list(range(0, len(text) - chunk_size + 1, stride))
                last_start = max(0, len(text) - chunk_size)
                if not starts or starts[-1] != last_start:
                    starts.append(last_start)
                windows = [text[s : s + chunk_size] for s in starts]
            total_chunks = len(windows)
            batch: List[Tuple[dict, str]] = []
            for chunk_index, window in enumerate(windows, start=1):
                metadata = {
                    "path": str(path),
                    "file_index": file_index,
                    "file_total": total_files,
                    "chunk_index": chunk_index,
                    "chunk_total": total_chunks,
                    "chunk_size": len(window),
                }
                batch.append((metadata, window))
                if len(batch) >= batch_cap:
                    yield batch
                    batch = []
            if batch:
                yield batch
