from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class AiderPolyglotAdapter(SimpleAggregateAdapter):
    """Evaluate solutions for the Aider Polyglot programming benchmark."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "Aider_Polyglot"
        self._config = config or AdapterRuntimeConfig.from_env()
        self._dataset_loader_override = dataset_loader
        self._completion_fn = completion_fn
        self._dataset: list[dict[str, object]] | None = None
        self._dataset_source = "uninitialized"

    def prepare(self, workdir: Path, seed: int) -> None:
        dataset = self._ensure_dataset()
        workdir.mkdir(parents=True, exist_ok=True)
        languages = sorted({str(item.get("language", "unknown")) for item in dataset})
        summary = {
            "examples": len(dataset),
            "languages": languages,
            "source": self._dataset_source,
            "options": {
                "dataset": self._config.get_option(self.name, "dataset", "Aider-AI/polyglot-benchmark"),
                "split": self._config.get_option(self.name, "split", "validation"),
            },
        }
        (workdir / "dataset_info.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._ensure_dataset()
        completion = self._resolve_completion_fn()

        predictions_path = output_dir / "predictions.jsonl"
        solved = 0
        total = 0
        with predictions_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(dataset):
                task_id = _coerce_task_id(record, index)
                prompt = _build_prompt(record, task_id)
                metadata = {
                    "task_id": task_id,
                    "language": record.get("language"),
                    "tests": record.get("tests"),
                }
                response = completion(prompt, model=model, seed=seed, metadata=metadata)
                evaluation = _evaluate_response(response, record)

                payload = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "response": response,
                    "checks": evaluation["checks"],
                    "solved": evaluation["solved"],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                total += 1
                if evaluation["solved"]:
                    solved += 1

        success_rate = float(solved) / total if total else 0.0
        metrics = {"success_rate": success_rate, "num_tasks": float(total)}
        metadata = {"solved": solved, "total": total}
        return MetricSummary(benchmark=self.name, seed=seed, metrics=metrics, metadata=metadata)

    def _resolve_completion_fn(self) -> CompletionFn:
        if self._completion_fn is None:
            self._completion_fn = resolve_completion_fn(self._config)
        return self._completion_fn

    def _ensure_dataset(self) -> list[dict[str, object]]:
        if self._dataset is not None:
            return self._dataset
        if self._dataset_loader_override is not None:
            raw_records = list(self._dataset_loader_override())
            self._dataset_source = "override"
        else:
            override_path = self._config.dataset_override(self.name)
            if override_path is not None:
                raw_records = load_local_records(override_path)
                self._dataset_source = f"local::{override_path}"
            else:
                raw_records = self._load_from_hub()
        self._dataset = [dict(record) for record in raw_records]
        logger.debug("Loaded %d Aider Polyglot examples (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "Aider-AI/polyglot-benchmark")
        split = self._config.get_option(self.name, "split", "validation")
        language = self._config.get_option(self.name, "language", None)
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for AiderPolyglotAdapter.") from exc

        load_args = {
            "split": split,
            "streaming": streaming,
            "token": self._config.hf_token,
            "cache_dir": str(self._config.hf_cache_dir),
        }
        if language:
            dataset = load_dataset(dataset_name, language, **load_args)
        else:
            dataset = load_dataset(dataset_name, **load_args)
        subset_label = language or "default"
        self._dataset_source = f"huggingface::{dataset_name}:{subset_label}:{split}"
        return list(dataset) if streaming else dataset


def _coerce_task_id(record: Mapping[str, object], fallback_index: int) -> str:
    for key in ("task_id", "problem_id", "id", "name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"aider_{fallback_index}"


def _build_prompt(record: Mapping[str, object], task_id: str) -> str:
    description = record.get("prompt") or record.get("description") or record.get("instruction")
    if description is None:
        description = json.dumps(record, ensure_ascii=False)
    language = record.get("language")
    tests = record.get("tests")
    instructions = [f"Task ID: {task_id}"]
    if language:
        instructions.append(f"Language: {language}")
    instructions.append("Implement the requested changes. Ensure the final answer only contains the updated source code or patch.")
    instructions.append("")
    instructions.append(str(description).strip())
    if tests:
        instructions.append("")
        instructions.append("Reference Tests:")
        if isinstance(tests, Sequence) and not isinstance(tests, (str, bytes)):
            for test in tests:
                instructions.append(f"- {test}")
        else:
            instructions.append(str(tests))
    instructions.append("")
    instructions.append("Solution:")
    return "\n".join(instructions)


def _evaluate_response(response: str, record: Mapping[str, object]) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    solved = True

    exact_targets = _as_list(record.get("expected_exact"))
    if exact_targets:
        normalized = _normalize_code(response)
        exact_pass = any(_normalize_code(target) == normalized for target in exact_targets)
        checks.append({"type": "exact", "passed": exact_pass})
        solved = solved and exact_pass

    snippet_targets = _as_list(record.get("expected_snippets"))
    if snippet_targets:
        snippet_pass = all(snippet in response for snippet in snippet_targets)
        checks.append({"type": "snippet", "passed": snippet_pass, "required": snippet_targets})
        solved = solved and snippet_pass

    regex_targets = _as_list(record.get("expected_regex"))
    for pattern in regex_targets:
        try:
            compiled = re.compile(str(pattern), re.MULTILINE)
        except re.error as exc:
            logger.warning("Invalid regex in record %s: %s", record.get("task_id"), exc)
            continue
        regex_pass = bool(compiled.search(response))
        checks.append({"type": "regex", "pattern": pattern, "passed": regex_pass})
        solved = solved and regex_pass

    if not checks:
        solved = False
        checks.append({"type": "reference", "passed": False, "reason": "No reference expectations provided."})

    return {"solved": solved, "checks": checks}


def _normalize_code(text: str) -> str:
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return "\n".join(lines)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        items = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
        return items
    return []
