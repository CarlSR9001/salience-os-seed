from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class GPQAAdapter(SimpleAggregateAdapter):
    """Adapter for the GPQA benchmark using multiple-choice evaluation."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "GPQA"
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
                "dataset": self._config.get_option(self.name, "dataset", "Idavidrein/gpqa"),
                "subset": self._config.get_option(self.name, "subset", "gpqa_diamond"),
                "split": self._config.get_option(self.name, "split", "validation"),
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
                question_id = _coerce_question_id(record, index)
                prompt, choices, answer_letter = _build_prompt_and_metadata(question_id, record)
                metadata = {"question_id": question_id, "choices": choices}
                response = completion(prompt, model=model, seed=seed, metadata=metadata)
                predicted_letter = _extract_letter(response, choices)
                is_correct = predicted_letter == answer_letter
                payload = {
                    "question_id": question_id,
                    "prompt": prompt,
                    "response": response,
                    "prediction": predicted_letter,
                    "answer": answer_letter,
                    "choices": choices,
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
        logger.debug("Loaded %d GPQA examples (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "Idavidrein/gpqa")
        subset = self._config.get_option(self.name, "subset", "gpqa_diamond")
        split = self._config.get_option(self.name, "split", "validation")
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for GPQAAdapter.") from exc
        if subset:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=streaming,
                token=self._config.hf_token,
                cache_dir=str(self._config.hf_cache_dir),
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                token=self._config.hf_token,
                cache_dir=str(self._config.hf_cache_dir),
            )
        self._dataset_source = f"huggingface::{dataset_name}:{subset or 'default'}:{split}"
        return list(dataset) if streaming else dataset


def _coerce_question_id(record: Mapping[str, object], fallback_index: int) -> str:
    for key in ("question_id", "id", "uid", "index"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"gpqa_{fallback_index}"


def _build_prompt_and_metadata(
    question_id: str, record: Mapping[str, object]
) -> tuple[str, dict[str, str], str]:
    question = record.get("question") or record.get("prompt") or record.get("query")
    if question is None:
        question = json.dumps(record, ensure_ascii=False)

    choices = _extract_choices(record)
    answer_letter = _determine_answer_letter(record, choices)

    instructions = (
        "Select the single best answer to the following graduate-level question."
        " Respond with the letter of the correct choice."
    )
    lines = [f"Question ID: {question_id}", instructions, "", str(question).strip(), ""]
    for letter, text in choices.items():
        lines.append(f"{letter}. {text}")
    lines.append("")
    lines.append("Answer:")
    prompt = "\n".join(lines)
    return prompt, choices, answer_letter


def _extract_choices(record: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if "choices" in record and isinstance(record["choices"], Mapping):
        for key, value in record["choices"].items():
            letter = _normalize_letter(key)
            normalized[letter] = str(value)
    elif "choices" in record and isinstance(record["choices"], Sequence):
        letters = list("ABCDE")
        for idx, value in enumerate(record["choices"]):
            if idx < len(letters):
                normalized[letters[idx]] = str(value)
    else:
        for letter in "ABCDE":
            candidate = record.get(letter) or record.get(letter.lower())
            if candidate is not None:
                normalized[_normalize_letter(letter)] = str(candidate)
    if not normalized:
        raise ValueError("GPQA record missing choices")
    return normalized


def _determine_answer_letter(record: Mapping[str, object], choices: Mapping[str, str]) -> str:
    answer = record.get("answer") or record.get("correct_answer") or record.get("label")
    if answer is None:
        raise ValueError("GPQA record missing answer key")
    if isinstance(answer, str):
        normalized = _normalize_letter(answer)
        if normalized in choices:
            return normalized
        # Some datasets store the answer text directly.
        for letter, text in choices.items():
            if normalized == _normalize_letter(text) or answer.strip() == text.strip():
                return letter
    if isinstance(answer, int):
        letters = list(choices.keys())
        if 0 <= answer < len(letters):
            return letters[answer]
    raise ValueError(f"Unable to determine correct letter from answer={answer!r}")


_LETTER_PATTERN = re.compile(r"[A-E]")


def _extract_letter(response: str, choices: Mapping[str, str]) -> str:
    match = _LETTER_PATTERN.search(response.upper())
    if match:
        return match.group(0)
    normalized_text = response.strip().lower()
    for letter, text in choices.items():
        if normalized_text == text.strip().lower():
            return letter
    return response.strip().upper()


def _normalize_letter(value: str) -> str:
    value = value.strip().upper()
    if value.startswith("OPTION "):
        value = value.split(" ", 1)[1]
    if value.endswith(")") and len(value) == 2:
        value = value[0]
    if value.startswith("(") and value.endswith(")") and len(value) == 3:
        value = value[1]
    return value[:1]
