from __future__ import annotations

import json
from pathlib import Path

import pytest

from salience_bench import __main__ as bench_cli
from salience_bench.adapters import REGISTRY
from salience_bench.adapters.aider_polyglot import AiderPolyglotAdapter
from salience_bench.adapters.aime_2024 import AIME2024Adapter
from salience_bench.adapters.bfcl import BFCLAdapter
from salience_bench.adapters.grind import GRINDAdapter
from salience_bench.adapters.swe_bench import SWEBenchAdapter
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


def test_bfcl_adapter_function_call(tmp_path, monkeypatch):
    dataset = [
        {
            "task_id": "weather_lookup",
            "instruction": "Call the weather tool for San Francisco in Celsius.",
            "tools": [
                {"name": "weather", "description": "Retrieve weather for a location", "parameters": {"location": "str"}}
            ],
            "expected_call": {
                "name": "weather",
                "arguments": {"location": "San Francisco", "units": "celsius"},
            },
        },
        {
            "task_id": "calendar_add",
            "instruction": "Schedule a meeting for tomorrow at 9am.",
            "tools": [
                {"name": "create_event", "description": "Add an event to the calendar"},
                {"name": "weather", "description": "Fallback weather API"},
            ],
            "expected_call": {
                "name": "create_event",
                "arguments": {"title": "Meeting", "time": "09:00", "day": "tomorrow"},
            },
        },
    ]

    answers = {item["task_id"]: item["expected_call"] for item in dataset}

    def loader():
        return dataset

    def completion(prompt: str, *, model: ModelConfig, seed: int, metadata: dict[str, object] | None = None) -> str:
        assert metadata is not None
        call = answers[str(metadata["task_id"])]
        return json.dumps(call)

    class FixtureBFCLAdapter(BFCLAdapter):
        def __init__(self) -> None:
            super().__init__(dataset_loader=loader, completion_fn=completion)

    monkeypatch.setitem(REGISTRY, "BFCL", FixtureBFCLAdapter)

    manifest = bench_cli.load_manifest(Path("benchmarks/benchmarks_manifest.json"))
    model_cfg = ModelConfig(name="fixture")

    bench_cli.run_benchmarks(
        ["BFCL"],
        manifest,
        seeds=[7],
        output=tmp_path,
        model_cfg=model_cfg,
        local_only=True,
    )

    run_root = next(tmp_path.iterdir())
    bench_dir = run_root / "BFCL"
    summary = json.loads((bench_dir / "summary.json").read_text())
    assert summary["mean"]["success_rate"] == pytest.approx(1.0)


def test_grind_adapter_reasoning(tmp_path, monkeypatch):
    dataset = [
        {
            "scenario_id": "logic_1",
            "context": "Alice is taller than Bob. Bob is taller than Carol.",
            "question": "Who is the tallest?",
            "choices": {"A": "Alice", "B": "Bob", "C": "Carol"},
            "answer": "A",
        },
        {
            "scenario_id": "logic_2",
            "context": "Paris is the capital of France.",
            "question": "Which city is the capital of France?",
            "answer": "Paris",
            "expected_regex": r"paris",
        },
    ]

    responses = {
        "logic_1": "A",
        "logic_2": "Paris is the capital of France.",
    }

    def loader():
        return dataset

    def completion(prompt: str, *, model: ModelConfig, seed: int, metadata: dict[str, object] | None = None) -> str:
        assert metadata is not None
        return responses[str(metadata["scenario_id"])]

    class FixtureGRINDAdapter(GRINDAdapter):
        def __init__(self) -> None:
            super().__init__(dataset_loader=loader, completion_fn=completion)

    monkeypatch.setitem(REGISTRY, "GRIND", FixtureGRINDAdapter)

    manifest = bench_cli.load_manifest(Path("benchmarks/benchmarks_manifest.json"))
    model_cfg = ModelConfig(name="fixture")

    bench_cli.run_benchmarks(
        ["GRIND"],
        manifest,
        seeds=[17],
        output=tmp_path,
        model_cfg=model_cfg,
        local_only=True,
    )

    run_root = next(tmp_path.iterdir())
    bench_dir = run_root / "GRIND"
    summary = json.loads((bench_dir / "summary.json").read_text())
    assert summary["mean"]["success_rate"] == pytest.approx(1.0)


def test_swe_bench_adapter_patch(tmp_path, monkeypatch):
    reference_patch = """diff --git a/app.py b/app.py\n@@\n-def add(a, b):\n-    return a - b\n+def add(a, b):\n+    return a + b\n"""

    dataset = [
        {
            "instance_id": "demo_1",
            "problem_statement": "Fix the add function so it sums its arguments.",
            "repo": "demo/app",
            "tests": ["pytest tests/test_app.py::test_add"],
            "expected_patch": reference_patch,
        }
    ]

    def loader():
        return dataset

    def completion(prompt: str, *, model: ModelConfig, seed: int, metadata: dict[str, object] | None = None) -> str:
        assert metadata is not None
        assert metadata["instance_id"] == "demo_1"
        return reference_patch

    class FixtureSWEBenchAdapter(SWEBenchAdapter):
        def __init__(self) -> None:
            super().__init__(dataset_loader=loader, completion_fn=completion)

    monkeypatch.setitem(REGISTRY, "SWE_bench", FixtureSWEBenchAdapter)

    manifest = bench_cli.load_manifest(Path("benchmarks/benchmarks_manifest.json"))
    model_cfg = ModelConfig(name="fixture")

    bench_cli.run_benchmarks(
        ["SWE_bench"],
        manifest,
        seeds=[5],
        output=tmp_path,
        model_cfg=model_cfg,
        local_only=True,
    )

    run_root = next(tmp_path.iterdir())
    bench_dir = run_root / "SWE_bench"
    summary = json.loads((bench_dir / "summary.json").read_text())
    assert summary["mean"]["success_rate"] == pytest.approx(1.0)


def test_aider_polyglot_adapter_snippets(tmp_path, monkeypatch):
    dataset = [
        {
            "task_id": "sum_numbers",
            "prompt": "Write a Python function `add_numbers(a, b)` that returns their sum.",
            "language": "python",
            "tests": ["assert add_numbers(2, 3) == 5"],
            "expected_snippets": ["def add_numbers(a, b):", "return a + b"],
        }
    ]

    answers = {
        "sum_numbers": "def add_numbers(a, b):\n    return a + b\n",
    }

    def loader():
        return dataset

    def completion(prompt: str, *, model: ModelConfig, seed: int, metadata: dict[str, object] | None = None) -> str:
        assert metadata is not None
        return answers[str(metadata["task_id"])]

    class FixtureAiderAdapter(AiderPolyglotAdapter):
        def __init__(self) -> None:
            super().__init__(dataset_loader=loader, completion_fn=completion)

    monkeypatch.setitem(REGISTRY, "Aider_Polyglot", FixtureAiderAdapter)

    manifest = bench_cli.load_manifest(Path("benchmarks/benchmarks_manifest.json"))
    model_cfg = ModelConfig(name="fixture")

    bench_cli.run_benchmarks(
        ["Aider_Polyglot"],
        manifest,
        seeds=[3],
        output=tmp_path,
        model_cfg=model_cfg,
        local_only=True,
    )

    run_root = next(tmp_path.iterdir())
    bench_dir = run_root / "Aider_Polyglot"
    summary = json.loads((bench_dir / "summary.json").read_text())
    assert summary["mean"]["success_rate"] == pytest.approx(1.0)
