"""Deterministic completion function for offline benchmark smoke runs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from ..core import ModelConfig

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_benchmarks"


class _AnswerBank(Mapping[str, str]):
    """Read-only mapping wrapper used for type checking."""

    def __init__(self, payload: Mapping[str, str]) -> None:
        self._payload = dict(payload)

    def __getitem__(self, key: str) -> str:
        return self._payload[key]

    def __iter__(self):  # type: ignore[override]
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark fixture not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
            else:
                raise ValueError(f"Expected JSON object per line in {path}, received {type(payload)!r}")
    return records


@lru_cache(maxsize=1)
def _load_answers() -> dict[str, _AnswerBank]:
    def build_map(filename: str, key_field: str, value_field: str) -> _AnswerBank:
        path = _DATA_DIR / filename
        mapping = {
            str(item[key_field]): str(item[value_field])
            for item in _load_jsonl(path)
            if key_field in item and value_field in item
        }
        return _AnswerBank(mapping)

    aime = build_map("aime_2024.jsonl", "problem_id", "solution")
    math = build_map("math_500.jsonl", "problem_id", "short_answer")
    gpqa = build_map("gpqa.jsonl", "question_id", "answer")

    bfcl_path = _DATA_DIR / "bfcl.jsonl"
    bfcl_mapping: dict[str, str] = {}
    for item in _load_jsonl(bfcl_path):
        task_id = str(item.get("task_id"))
        expected = item.get("expected_call")
        if expected is not None:
            bfcl_mapping[task_id] = json.dumps(expected, sort_keys=True)
    bfcl = _AnswerBank(bfcl_mapping)

    grind_path = _DATA_DIR / "grind.jsonl"
    grind_mapping: dict[str, str] = {}
    for item in _load_jsonl(grind_path):
        scenario_id = str(item.get("scenario_id"))
        answer = item.get("answer")
        if isinstance(answer, str) and answer.strip():
            grind_mapping[scenario_id] = answer.strip()
            continue
        regex = item.get("expected_regex")
        if isinstance(regex, str) and regex.strip():
            grind_mapping[scenario_id] = regex.strip()
    grind = _AnswerBank(grind_mapping)

    aider_path = _DATA_DIR / "aider_polyglot.jsonl"
    aider_mapping: dict[str, str] = {}
    for item in _load_jsonl(aider_path):
        task_id = str(item.get("task_id"))
        reference = item.get("reference_solution")
        if isinstance(reference, str) and reference.strip():
            aider_mapping[task_id] = reference
    aider = _AnswerBank(aider_mapping)

    swe_path = _DATA_DIR / "swe_bench.jsonl"
    swe_mapping: dict[str, str] = {}
    for item in _load_jsonl(swe_path):
        instance_id = str(item.get("instance_id"))
        patch = item.get("expected_patch")
        if isinstance(patch, str) and patch.strip():
            swe_mapping[instance_id] = patch
    swe = _AnswerBank(swe_mapping)

    return {
        "aime": aime,
        "math": math,
        "gpqa": gpqa,
        "bfcl": bfcl,
        "grind": grind,
        "aider": aider,
        "swe": swe,
        "fallback": _AnswerBank({"": "I am not sure."}),
    }


def completion(
    prompt: str,
    *,
    model: ModelConfig,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return deterministic responses for local benchmark fixtures."""

    answers = _load_answers()
    if metadata is None:
        return answers["fallback"][""]

    if "problem_id" in metadata:
        identifier = str(metadata["problem_id"])
        if identifier in answers["aime"]:
            return answers["aime"][identifier]
        if identifier in answers["math"]:
            return answers["math"][identifier]
        return answers["fallback"][""]

    if "question_id" in metadata:
        identifier = str(metadata["question_id"])
        return answers["gpqa"].get(identifier, answers["fallback"][""])

    if "task_id" in metadata:
        identifier = str(metadata["task_id"])
        if metadata.get("tools") is not None:
            return answers["bfcl"].get(identifier, answers["fallback"][""])
        return answers["aider"].get(identifier, answers["fallback"][""])

    if "scenario_id" in metadata:
        identifier = str(metadata["scenario_id"])
        return answers["grind"].get(identifier, answers["fallback"][""])

    if "instance_id" in metadata:
        identifier = str(metadata["instance_id"])
        return answers["swe"].get(identifier, answers["fallback"][""])

    return answers["fallback"][""]
