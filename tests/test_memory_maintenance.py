import json
from pathlib import Path

from salience_os_seed.core.memory import StructuredMemory
from salience_os_seed.core.memory.maintenance import (
    ArchiveStore,
    MaintenanceThresholds,
    archive_low_roi_facts,
    merge_redundant_entries,
    prune_failed_hypotheses,
    should_cleanup,
    summarize_old_context,
)


def test_should_cleanup_triggers_on_table_size():
    memory = StructuredMemory()
    for idx in range(12):
        memory.facts.add(f"fact {idx}")
    thresholds = MaintenanceThresholds(facts_max=10, drag_trigger=0.9)
    assert should_cleanup({"drag": 0.1}, memory, thresholds)


def test_archive_low_roi_facts_writes_and_prunes(tmp_path: Path):
    memory = StructuredMemory()
    memory.facts.add("retain", metadata={"roi": 0.5})
    memory.facts.add("archive", metadata={"roi": 0.05})
    archive_root = tmp_path / "archive"
    store = ArchiveStore(root=archive_root)

    archived = archive_low_roi_facts(memory, threshold=0.1, store=store)

    assert archived == 1
    assert memory.facts.count() == 1
    files = list(archive_root.glob("facts_*.jsonl"))
    assert files, "expected archive file to be created"
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["text"] == "archive"


def test_merge_and_prune_cleanup():
    memory = StructuredMemory()
    memory.hypotheses.add("duplicate entry", metadata={"failures": 2})
    memory.hypotheses.add("duplicate entry", metadata={"failures": 4})
    merged = merge_redundant_entries(memory, similarity_threshold=0.5)
    assert merged == 1
    removed = prune_failed_hypotheses(memory, failure_limit=3)
    assert removed == 1
    assert memory.hypotheses.count() == 0


def test_summarize_old_context_rolls_up_records():
    memory = StructuredMemory()
    for idx in range(5):
        memory.todos.add(f"task {idx}")
    summarized = summarize_old_context(memory, age_threshold=1, chunk_size=2)
    assert summarized == 5
    summaries = [record.text for record in memory.todos.iter()]
    assert any(text.startswith("todos_summary::") for text in summaries)
