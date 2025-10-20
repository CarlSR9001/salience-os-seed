from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class BFCLAdapter(SimpleAggregateAdapter):
    """Evaluate function-calling tasks from the Berkeley Function-Calling Leaderboard."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "BFCL"
        self._config = config or AdapterRuntimeConfig.from_env()
        self._dataset_loader_override = dataset_loader
        self._completion_fn = completion_fn
        self._dataset: list[dict[str, object]] | None = None
        self._dataset_source = "uninitialized"

    def prepare(self, workdir: Path, seed: int) -> None:
        dataset = self._ensure_dataset()
        workdir.mkdir(parents=True, exist_ok=True)
        tools = sorted({tool["name"] for record in dataset for tool in _as_tool_list(record.get("tools"))})
        summary = {
            "examples": len(dataset),
            "source": self._dataset_source,
            "tools": tools,
            "options": {
                "dataset": self._config.get_option(self.name, "dataset", "gorilla-llm/BFCL"),
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
                metadata = {"task_id": task_id, "tools": record.get("tools")}
                response = completion(prompt, model=model, seed=seed, metadata=metadata)
                evaluation = _evaluate_function_call(response, record)

                payload = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "response": response,
                    "parsed_call": evaluation.get("parsed_call"),
                    "solved": evaluation["solved"],
                    "checks": evaluation["checks"],
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
        logger.debug("Loaded %d BFCL tasks (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "gorilla-llm/BFCL")
        split = self._config.get_option(self.name, "split", "validation")
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for BFCLAdapter.") from exc

        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            token=self._config.hf_token,
            cache_dir=str(self._config.hf_cache_dir),
        )
        self._dataset_source = f"huggingface::{dataset_name}:{split}"
        return list(dataset) if streaming else dataset


def _coerce_task_id(record: Mapping[str, object], fallback_index: int) -> str:
    for key in ("task_id", "id", "name", "uid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"bfcl_{fallback_index}"


def _build_prompt(record: Mapping[str, object], task_id: str) -> str:
    instruction = record.get("instruction") or record.get("prompt") or record.get("query")
    if instruction is None:
        instruction = json.dumps(record, ensure_ascii=False)
    lines = [f"Task ID: {task_id}", "You are an assistant with access to structured tools."]
    context = record.get("context")
    if context:
        lines.extend(["", str(context).strip()])
    lines.extend(["", "Available tools:"])
    tools = _as_tool_list(record.get("tools"))
    if not tools:
        lines.append("- (none provided)")
    for tool in tools:
        lines.append(f"- {tool['name']}: {tool.get('description', '')}")
    lines.extend([
        "",
        "Instruction:",
        str(instruction).strip(),
        "",
        "Respond with a JSON object describing the tool call, including 'name' and 'arguments'.",
    ])
    return "\n".join(lines)


def _evaluate_function_call(response: str, record: Mapping[str, object]) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    solved = True

    expected = record.get("expected_call") or record.get("reference_call")
    parsed_call = None
    if expected is None:
        solved = False
        checks.append({"type": "reference", "passed": False, "reason": "Missing expected_call in record."})
    else:
        try:
            parsed_call = _parse_function_call(response)
            checks.append({"type": "parse", "passed": True})
        except ValueError as exc:
            solved = False
            checks.append({"type": "parse", "passed": False, "reason": str(exc)})
            parsed_call = None

        if parsed_call is not None:
            expected_name = str(expected.get("name"))
            name_pass = parsed_call["name"] == expected_name
            checks.append({"type": "name", "passed": name_pass, "expected": expected_name})
            solved = solved and name_pass

            expected_args = _normalize_arguments(expected.get("arguments", {}))
            arg_pass = parsed_call["arguments"] == expected_args
            checks.append({"type": "arguments", "passed": arg_pass, "expected": expected_args})
            solved = solved and arg_pass

    optional = record.get("expected_response")
    if optional is not None:
        expected_texts = _as_string_list(optional)
        if expected_texts:
            match = any(text in response for text in expected_texts)
            checks.append({"type": "response", "passed": match, "expected": expected_texts})
            solved = solved and match

    return {"solved": solved, "checks": checks, "parsed_call": parsed_call}


def _parse_function_call(response: str) -> dict[str, object]:
    text = response.strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        # Strip optional code fences.
        parts = text.split("\n", 1)
        if len(parts) == 2:
            text = parts[1]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping) and "function_call" in payload:
        payload = payload["function_call"]

    if not isinstance(payload, Mapping):
        raise ValueError("Parsed payload must be an object")

    name = payload.get("name")
    if name is None:
        raise ValueError("Function call missing 'name'")

    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Arguments field is not valid JSON: {exc}") from exc

    if not isinstance(arguments, Mapping):
        raise ValueError("Function call arguments must be a JSON object")

    return {"name": str(name), "arguments": _normalize_arguments(arguments)}


def _normalize_arguments(obj: object) -> object:
    if isinstance(obj, Mapping):
        return {str(key): _normalize_arguments(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_normalize_arguments(item) for item in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def _as_tool_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    tools: list[Mapping[str, object]] = []
    for item in value:
        if isinstance(item, Mapping) and "name" in item:
            tools.append(item)
    return tools


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []
