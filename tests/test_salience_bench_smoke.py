from __future__ import annotations

import json
from pathlib import Path

import pytest

from salience_bench import __main__ as bench_cli
from salience_bench.adapters import REGISTRY
from salience_bench.adapters.aime_2024 import AIME2024Adapter
from salience_bench.core import ModelConfig


def test_aime_adapter_end_to_end(tmp_path, monkeypatch):
    dataset = [
        {"problem_id": "A1", "problem": "What is 1 + 1?", "solution": "2"},
        {"problem_id": "A2", "problem": "Compute 2 + 2.", "solution": "4"},
    ]
    answers = {item["problem_id"]: item["solution"] for item in dataset}

    def loader():
        return dataset

    def completion(prompt: str, *, model: ModelConfig, seed: int, metadata: dict[str, object] | None = None) -> str:
        assert metadata is not None
        return answers[str(metadata["problem_id"])]

    class FixtureAIMEAdapter(AIME2024Adapter):
        def __init__(self) -> None:
            super().__init__(dataset_loader=loader, completion_fn=completion)

    monkeypatch.setitem(REGISTRY, "AIME_2024", FixtureAIMEAdapter)

    manifest = bench_cli.load_manifest(Path("benchmarks/benchmarks_manifest.json"))
    model_cfg = ModelConfig(name="fixture")
    seeds = [11, 22]

    bench_cli.run_benchmarks(
        ["AIME_2024"],
        manifest,
        seeds=seeds,
        output=tmp_path,
        model_cfg=model_cfg,
        local_only=True,
    )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_root = run_dirs[0]

    combined = json.loads((run_root / "combined_report.json").read_text())
    assert combined["aggregates"][0]["mean"]["accuracy"] == pytest.approx(1.0)

    bench_dir = run_root / "AIME_2024"
    summary = json.loads((bench_dir / "summary.json").read_text())
    assert summary["mean"]["accuracy"] == pytest.approx(1.0)

    for seed in seeds:
        seed_dir = bench_dir / f"seed_{seed}"
        assert (seed_dir / "predictions.jsonl").exists()
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        assert metrics["metrics"]["accuracy"] == pytest.approx(1.0)
        assert metrics["metadata"]["correct"] == 2
