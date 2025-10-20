from __future__ import annotations

import json
import logging
import re
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class MATH500Adapter(SimpleAggregateAdapter):
    """Adapter for the MATH-500 benchmark using short-answer evaluation."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "MATH_500"
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
                "dataset": self._config.get_option(self.name, "dataset", "HuggingFaceH4/MATH-500"),
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
                prediction = _normalize_expression(response)
                references = _extract_references(record)
                normalized_refs = [_normalize_expression(value) for value in references]
                is_correct = prediction in normalized_refs
                payload = {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "response": response,
                    "normalized_prediction": prediction,
                    "references": references,
                    "normalized_references": normalized_refs,
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
        logger.debug("Loaded %d MATH-500 examples (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "HuggingFaceH4/MATH-500")
        split = self._config.get_option(self.name, "split", "test")
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for MATH500Adapter.") from exc
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
    return f"math500_{fallback_index}"


def _build_prompt(problem_id: str, record: Mapping[str, object]) -> str:
    statement = record.get("problem") or record.get("question") or record.get("prompt")
    if statement is None:
        statement = json.dumps(record, ensure_ascii=False)
    instructions = (
        "Solve the following mathematics competition problem. Reply with only the final numeric or algebraic answer."
    )
    return f"Problem ID: {problem_id}\n{instructions}\n\n{statement}\nAnswer:"


def _extract_references(record: Mapping[str, object]) -> list[str]:
    candidates: list[str] = []
    for key in ("short_answer", "answers", "answer", "solution", "solutions"):
        value = record.get(key)
        if isinstance(value, str):
            candidates.extend(_split_multi_answer(value))
        elif isinstance(value, Sequence):
            for item in value:
                if isinstance(item, str):
                    candidates.extend(_split_multi_answer(item))
    if not candidates and "solution" in record:
        text = str(record["solution"])
        matches = re.findall(r"\\boxed\{([^}]*)\}", text)
        if matches:
            candidates.extend(match.strip() for match in matches)
    if not candidates:
        logger.debug("Falling back to raw solution text for problem %s", record.keys())
        text = str(record.get("solution", record))
        candidates.append(text)
    return candidates


def _split_multi_answer(text: str) -> list[str]:
    if ";" in text:
        return [segment.strip() for segment in text.split(";") if segment.strip()]
    return [text.strip()]


def _normalize_expression(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\\boxed{", "").replace("}", "")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = cleaned.replace("−", "-")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("\n", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("*", "\u00d7") if cleaned.count("*") == 1 else cleaned
    fraction_match = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", cleaned)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        frac = Fraction(numerator, denominator)
        return f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator)
    try:
        frac = Fraction(cleaned)
        if frac.denominator != 1:
            return f"{frac.numerator}/{frac.denominator}"
        return str(frac.numerator)
    except Exception:
        pass
    return cleaned.lower()
