from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class GRINDAdapter(SimpleAggregateAdapter):
    """Evaluate the Vellum GRIND grounded reasoning benchmark."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "GRIND"
        self._config = config or AdapterRuntimeConfig.from_env()
        self._dataset_loader_override = dataset_loader
        self._completion_fn = completion_fn
        self._dataset: list[dict[str, object]] | None = None
        self._dataset_source = "uninitialized"

    def prepare(self, workdir: Path, seed: int) -> None:
        dataset = self._ensure_dataset()
        workdir.mkdir(parents=True, exist_ok=True)
        domains = sorted({str(item.get("domain", "unknown")) for item in dataset})
        summary = {
            "examples": len(dataset),
            "source": self._dataset_source,
            "domains": domains,
            "options": {
                "dataset": self._config.get_option(self.name, "dataset", "vellumai/grind"),
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
                scenario_id = _coerce_scenario_id(record, index)
                prompt = _build_prompt(record, scenario_id)
                metadata = {
                    "scenario_id": scenario_id,
                    "domain": record.get("domain"),
                    "choices": record.get("choices"),
                }
                response = completion(prompt, model=model, seed=seed, metadata=metadata)
                evaluation = _evaluate_response(response, record)

                payload = {
                    "scenario_id": scenario_id,
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
        metrics = {"success_rate": success_rate, "num_questions": float(total)}
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
        logger.debug("Loaded %d GRIND scenarios (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "vellumai/grind")
        split = self._config.get_option(self.name, "split", "validation")
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for GRINDAdapter.") from exc

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            token=self._config.hf_token,
            cache_dir=str(self._config.hf_cache_dir),
        )
        self._dataset_source = f"huggingface::{dataset_name}:{split}"
        return list(dataset) if streaming else dataset


def _coerce_scenario_id(record: Mapping[str, object], fallback_index: int) -> str:
    for key in ("scenario_id", "id", "uid", "name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"grind_{fallback_index}"


def _build_prompt(record: Mapping[str, object], scenario_id: str) -> str:
    context = record.get("context") or record.get("passage") or record.get("document")
    question = record.get("question") or record.get("prompt")
    choices = record.get("choices")
    instructions = [
        f"Scenario ID: {scenario_id}",
        "Read the context carefully and provide the single best answer. Justify briefly if required.",
    ]
    if record.get("domain"):
        instructions.append(f"Domain: {record['domain']}")
    instructions.append("")
    if context:
        instructions.extend(["Context:", str(context).strip(), ""])
    if question:
        instructions.extend(["Question:", str(question).strip(), ""])
    if choices and isinstance(choices, Mapping):
        instructions.append("Choices:")
        for key, value in choices.items():
            instructions.append(f"{key}. {value}")
        instructions.append("")
        instructions.append("Answer with the letter of the correct choice.")
    instructions.append("")
    instructions.append("Answer:")
    return "\n".join(instructions)


def _evaluate_response(response: str, record: Mapping[str, object]) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    solved = True

    expected_answers = _extract_expected_answers(record)
    normalized_response = _normalize_text(response)
    if expected_answers:
        answer_pass = any(
            normalized_response == target or target in normalized_response for target in expected_answers
        )
        checks.append({"type": "answer", "passed": answer_pass, "expected": expected_answers})
        solved = solved and answer_pass

    required_keywords = _as_string_list(record.get("required_keywords"))
    if required_keywords:
        keyword_pass = all(keyword.lower() in response.lower() for keyword in required_keywords)
        checks.append({"type": "required_keywords", "passed": keyword_pass, "keywords": required_keywords})
        solved = solved and keyword_pass

    forbidden_keywords = _as_string_list(record.get("forbidden_keywords"))
    if forbidden_keywords:
        forbidden_pass = all(keyword.lower() not in response.lower() for keyword in forbidden_keywords)
        checks.append({"type": "forbidden_keywords", "passed": forbidden_pass, "keywords": forbidden_keywords})
        solved = solved and forbidden_pass

    regex_targets = _as_string_list(record.get("expected_regex"))
    for pattern in regex_targets:
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            logger.warning("Invalid regex for GRIND record %s: %s", record.get("scenario_id"), exc)
            continue
        regex_pass = bool(compiled.search(response))
        checks.append({"type": "regex", "pattern": pattern, "passed": regex_pass})
        solved = solved and regex_pass

    if not checks:
        solved = False
        checks.append({"type": "reference", "passed": False, "reason": "No evaluation criteria provided."})

    return {"solved": solved, "checks": checks}


def _extract_expected_answers(record: Mapping[str, object]) -> list[str]:
    candidates: list[str] = []
    for key in ("answer", "answers", "expected_answers", "label"):
        value = record.get(key)
        if isinstance(value, str):
            candidates.append(_normalize_text(value))
        elif isinstance(value, Sequence):
            for item in value:
                if isinstance(item, str):
                    candidates.append(_normalize_text(item))
    if not candidates and record.get("choices"):
        # Some datasets specify answer index.
        index = record.get("answer_index")
        if isinstance(index, int):
            keys = list(record["choices"].keys()) if isinstance(record["choices"], Mapping) else list(record["choices"])
            if 0 <= index < len(keys):
                candidates.append(_normalize_text(str(keys[index])))
    return candidates


def _normalize_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []
