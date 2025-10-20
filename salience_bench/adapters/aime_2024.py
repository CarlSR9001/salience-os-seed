from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class AIME2024Adapter(SimpleAggregateAdapter):
    """Evaluate models on the AIME 2024 short-answer mathematics benchmark."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "AIME_2024"
        self._config = config or AdapterRuntimeConfig.from_env()
        self._dataset_loader_override = dataset_loader
        self._completion_fn = completion_fn
        self._dataset: list[dict[str, object]] | None = None
        self._dataset_source = "uninitialized"

    def prepare(self, workdir: Path, seed: int) -> None:
        dataset = self._ensure_dataset()
        workdir.mkdir(parents=True, exist_ok=True)
        summary = {
            "examples": len(dataset),
            "source": self._dataset_source,
            "options": {
                "dataset": self._config.get_option(self.name, "dataset", "math-ai/aime24"),
                "split": self._config.get_option(self.name, "split", "test"),
            },
        }
        (workdir / "dataset_info.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._ensure_dataset()
        completion = self._resolve_completion_fn()

        predictions_path = output_dir / "predictions.jsonl"
        correct = 0
        total = 0
        with predictions_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(dataset):
                problem_id = _coerce_problem_id(record, index)
                prompt = _build_prompt(problem_id, record)
                metadata = {"problem_id": problem_id, "seed": seed}
                response = completion(prompt, model=model, seed=seed, metadata=metadata)
                normalized_prediction = _normalize_answer(response)
                references = _extract_references(record)
                normalized_references = [_normalize_answer(value) for value in references]
                is_correct = normalized_prediction in normalized_references
                payload = {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "response": response,
                    "normalized_prediction": normalized_prediction,
                    "references": references,
                    "normalized_references": normalized_references,
                    "correct": is_correct,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                total += 1
                if is_correct:
                    correct += 1

        accuracy = float(correct) / total if total else 0.0
        metrics = {"accuracy": accuracy, "num_questions": float(total)}
        metadata = {"correct": correct, "total": total}
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
        logger.debug("Loaded %d AIME 2024 examples (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "math-ai/aime24")
        split = self._config.get_option(self.name, "split", "test")
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for AIME2024Adapter.") from exc
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            token=self._config.hf_token,
            cache_dir=str(self._config.hf_cache_dir),
        )
        self._dataset_source = f"huggingface::{dataset_name}:{split}"
        return list(dataset) if streaming else dataset


def _coerce_problem_id(record: Mapping[str, object], fallback_index: int) -> str:
    for key in ("problem_id", "id", "uid", "index"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"aime2024_{fallback_index}"


def _build_prompt(problem_id: str, record: Mapping[str, object]) -> str:
    statement = record.get("problem") or record.get("question") or record.get("prompt")
    if statement is None:
        statement = json.dumps(record, ensure_ascii=False)
    instructions = (
        "You are solving an AIME mathematics question. Provide only the final numeric answer as a single number."
    )
    return f"Problem ID: {problem_id}\n{instructions}\n\n{statement}\nAnswer:"


def _extract_references(record: Mapping[str, object]) -> list[str]:
    candidates: list[str] = []
    for key in ("solution", "solutions", "answer", "answers", "short_answer"):
        value = record.get(key)
        if isinstance(value, str):
            candidates.extend(_split_multi_answer(value))
        elif isinstance(value, Sequence):
            for item in value:
                if isinstance(item, str):
                    candidates.extend(_split_multi_answer(item))
    if not candidates:
        logger.debug("Falling back to raw record for reference extraction: keys=%s", list(record))
        text = str(record)
        candidates.append(text)
    return candidates


def _split_multi_answer(text: str) -> list[str]:
    # Allow semi-colon or comma separated enumerations but avoid splitting decimal numbers.
    if ";" in text:
        return [segment.strip() for segment in text.split(";") if segment.strip()]
    return [text.strip()]


_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize_answer(text: str) -> str:
    cleaned = text.strip()
    match = _BOXED_PATTERN.search(cleaned)
    if match:
        cleaned = match.group(1)
    cleaned = cleaned.replace("\\", "")
    cleaned = cleaned.replace("−", "-")
    cleaned = cleaned.strip()
    # Collapse whitespace and punctuation typically irrelevant for short answers.
    cleaned = re.sub(r"[\s\n]+", "", cleaned)
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.lower()
    return cleaned
