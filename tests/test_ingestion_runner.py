import json
from pathlib import Path

from salience_os_seed.ingestion.run import ingest_directory, main


def test_ingest_directory_returns_stats(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "sample.txt").write_text("hello world", encoding="utf-8")

    stats = ingest_directory(str(corpus_root), progress=False)
    assert stats["files_processed"] == 1
    assert stats["chunks_processed"] >= 1


def test_ingestion_main_writes_output(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "sample.txt").write_text("hello again", encoding="utf-8")
    output = tmp_path / "stats.json"

    exit_code = main([
        str(corpus_root),
        "--output",
        str(output),
        "--quiet",
        "--no-progress",
    ])

    assert exit_code == 0
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["files_processed"] == 1
