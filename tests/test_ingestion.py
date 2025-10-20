from pathlib import Path
from typing import Mapping, Tuple

import pytest

from salience_os_seed.conversation.session import ConversationConfig, ConversationSession, IngestionConfig


def test_ingestion_loop_filters_progress_and_autosaves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.txt").write_text("first chunk", encoding="utf-8")
    (corpus_root / "b.txt").write_text("second chunk", encoding="utf-8")

    autosave_path = tmp_path / "autosave.json"
    config = ConversationConfig(
        auto_save_path=str(autosave_path),
        ingestion=IngestionConfig(checkpoint_interval=1, max_chunks=2),
    )
    session = ConversationSession(config=config)

    trained_chunks: list[str] = []

    def fake_training_step(chunk: str) -> None:
        trained_chunks.append(chunk)

    monkeypatch.setattr(session.proto_lm, "training_step", fake_training_step)

    decision_iter = iter(
        [
            (True, {"uncertainty": 0.9, "novelty": 0.8}),
            (False, {"uncertainty": 0.1, "novelty": 0.1}),
        ]
    )

    def fake_evaluate(
        state: Mapping[str, object],
        memory_snapshot: Mapping[str, object],
        meta_snapshot: Mapping[str, float],
    ) -> Tuple[bool, Mapping[str, float]]:
        try:
            return next(decision_iter)
        except StopIteration:
            return False, {"uncertainty": 0.0}

    monkeypatch.setattr(session._filter, "evaluate", fake_evaluate)

    progress_payloads: list[Mapping[str, object]] = []

    def progress_cb(payload: Mapping[str, object]) -> None:
        progress_payloads.append(payload)

    stats = session.ingest_directory(str(corpus_root), progress_cb=progress_cb)

    assert stats.chunks_processed == 2
    assert stats.chunks_accepted == 1
    assert stats.chunks_rejected == 1
    assert len(trained_chunks) == 1
    assert autosave_path.exists()
    assert len(progress_payloads) == 2
    assert progress_payloads[0]["accepted"] is True
    assert "readings" in progress_payloads[0]


def test_ingestion_overlap_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "win.txt").write_text("abcdefghij", encoding="utf-8")

    config = ConversationConfig(
        ingestion=IngestionConfig(
            chunk_size=4,
            chunk_overlap=2,
            batch_size=2,
            checkpoint_interval=10,
        )
    )
    session = ConversationSession(config=config)

    accepted_chunks: list[str] = []

    def fake_training(chunk: str) -> None:
        accepted_chunks.append(chunk)

    monkeypatch.setattr(session.proto_lm, "training_step", fake_training)

    def always_accept(*_args, **_kwargs):
        return True, {"uncertainty": 1.0}

    monkeypatch.setattr(session._filter, "evaluate", always_accept)

    stats = session.ingest_directory(str(corpus_root))

    assert stats.chunks_processed == 4
    assert stats.chunks_accepted == 4
    assert stats.chunks_rejected == 0
    assert accepted_chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_ingestion_respects_learning_toggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "x.txt").write_text("hello world", encoding="utf-8")

    config = ConversationConfig(
        learning_enabled=False,
        ingestion=IngestionConfig(checkpoint_interval=1),
    )
    session = ConversationSession(config=config)

    trained: list[str] = []

    def recorder(chunk: str) -> None:
        trained.append(chunk)

    monkeypatch.setattr(session.proto_lm, "training_step", recorder)

    def always_accept(*_args, **_kwargs):
        return True, {}

    monkeypatch.setattr(session._filter, "evaluate", always_accept)

    stats = session.ingest_directory(str(corpus_root))

    assert stats.chunks_processed == 1
    assert stats.chunks_accepted == 1
    assert trained == []
