"""Ingestion deduplication utilities backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, MutableMapping, Optional


DEFAULT_DB_PATH = Path("storage/ingestion/index.db")
_LOCK = threading.Lock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_digests (
            digest TEXT PRIMARY KEY,
            doc_name TEXT NOT NULL,
            length INTEGER NOT NULL,
            metadata TEXT
        )
        """
    )
    conn.commit()


@dataclass
class IngestRecord:
    digest: str
    doc_name: str
    length: int
    metadata: MutableMapping[str, object]


class IngestIndex:
    """SQLite-backed store for ingested content digests."""

    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread False to let multiple handlers share the connection;
        # a global lock ensures serialized access.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with _LOCK:
            _ensure_schema(self._conn)

    def seen(self, digest: str) -> bool:
        with _LOCK:
            cur = self._conn.execute(
                "SELECT 1 FROM ingestion_digests WHERE digest = ?",
                (digest,),
            )
            return cur.fetchone() is not None

    def fetch(self, digest: str) -> Optional[IngestRecord]:
        with _LOCK:
            cur = self._conn.execute(
                "SELECT digest, doc_name, length, metadata FROM ingestion_digests WHERE digest = ?",
                (digest,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        metadata = json.loads(row[3]) if row[3] else {}
        return IngestRecord(digest=row[0], doc_name=row[1], length=row[2], metadata=metadata)

    def mark(
        self,
        digest: str,
        *,
        doc_name: str,
        length: int,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        payload = json.dumps(dict(metadata or {}))
        with _LOCK:
            self._conn.execute(
                "INSERT OR IGNORE INTO ingestion_digests (digest, doc_name, length, metadata) VALUES (?, ?, ?, ?)",
                (digest, doc_name, length, payload),
            )
            self._conn.commit()

    def close(self) -> None:
        with _LOCK:
            self._conn.close()


def content_digest(payload: str | bytes | bytearray) -> str:
    if isinstance(payload, str):
        data = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")
    return sha256(data).hexdigest()


def ingest_once(
    doc_name: str,
    payload: str,
    index: IngestIndex,
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    digest = content_digest(payload)
    if index.seen(digest):
        return {
            "status": "skip",
            "reason": "duplicate",
            "digest": digest,
        }
    index.mark(digest, doc_name=doc_name, length=len(payload), metadata=metadata)
    return {
        "status": "ingested",
        "digest": digest,
    }
