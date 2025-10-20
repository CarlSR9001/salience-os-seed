from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from ..core import MetricSummary, ModelConfig, SimpleAggregateAdapter
from .base import AdapterRuntimeConfig, CompletionFn, load_local_records, resolve_completion_fn

logger = logging.getLogger(__name__)


class SWEBenchAdapter(SimpleAggregateAdapter):
    """Evaluate SWE-bench style repository repair tasks."""

    def __init__(
        self,
        *,
        dataset_loader: Callable[[], Iterable[Mapping[str, object]]] | None = None,
        completion_fn: CompletionFn | None = None,
        verifier: Callable[[Mapping[str, object], str], tuple[bool, list[dict[str, object]]]] | None = None,
        config: AdapterRuntimeConfig | None = None,
    ) -> None:
        self.name = "SWE_bench"
        self._config = config or AdapterRuntimeConfig.from_env()
        self._dataset_loader_override = dataset_loader
        self._completion_fn = completion_fn
        self._verifier = verifier
        self.requires_cloud = False
        self._dataset: list[dict[str, object]] | None = None
        self._dataset_source = "uninitialized"

    def prepare(self, workdir: Path, seed: int) -> None:
        dataset = self._ensure_dataset()
        workdir.mkdir(parents=True, exist_ok=True)
        repos = sorted({str(item.get("repo", "unknown")) for item in dataset})
        summary = {
            "examples": len(dataset),
            "source": self._dataset_source,
            "repos": repos,
            "options": {
                "dataset": self._config.get_option(self.name, "dataset", "princeton-nlp/SWE-bench_Lite"),
                "split": self._config.get_option(self.name, "split", "dev"),
                "subset": self._config.get_option(self.name, "subset", "default"),
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
                instance_id = _coerce_instance_id(record, index)
                prompt = _build_prompt(record, instance_id)
                metadata = {
                    "instance_id": instance_id,
                    "repo": record.get("repo"),
                    "tests": record.get("tests"),
                }
                response = completion(prompt, model=model, seed=seed, metadata=metadata)
                solved_flag, checks = self._verify_response(record, response)

                payload = {
                    "instance_id": instance_id,
                    "prompt": prompt,
                    "response": response,
                    "checks": checks,
                    "solved": solved_flag,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                total += 1
                if solved_flag:
                    solved += 1

        success_rate = float(solved) / total if total else 0.0
        metrics = {"success_rate": success_rate, "num_instances": float(total)}
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
        logger.debug("Loaded %d SWE-bench instances (%s)", len(self._dataset), self._dataset_source)
        return self._dataset

    def _load_from_hub(self) -> Sequence[Mapping[str, object]]:
        dataset_name = self._config.get_option(self.name, "dataset", "princeton-nlp/SWE-bench_Lite")
        subset = self._config.get_option(self.name, "subset", "default")
        split = self._config.get_option(self.name, "split", "dev")
        streaming_flag = self._config.get_option(self.name, "streaming", "false")
        streaming = isinstance(streaming_flag, str) and streaming_flag.lower() in {"1", "true", "yes"}
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The 'datasets' package is required for SWEBenchAdapter.") from exc

        load_args = {
            "split": split,
            "streaming": streaming,
            "token": self._config.hf_token,
            "cache_dir": str(self._config.hf_cache_dir),
        }
        if subset and subset != "default":
            dataset = load_dataset(dataset_name, subset, **load_args)
            subset_label = subset
        else:
            dataset = load_dataset(dataset_name, **load_args)
            subset_label = "default"
        self._dataset_source = f"huggingface::{dataset_name}:{subset_label}:{split}"
        return list(dataset) if streaming else dataset

    def _verify_response(self, record: Mapping[str, object], response: str) -> tuple[bool, list[dict[str, object]]]:
        if self._verifier is not None:
            return self._verifier(record, response)

        checks: list[dict[str, object]] = []
        solved = True

        expected_patch = _extract_patch(record)
        if expected_patch is not None:
            normalized_response = _normalize_patch(response)
            patch_pass = normalized_response == expected_patch
            checks.append({"type": "patch", "passed": patch_pass})
            solved = solved and patch_pass

        expected_outcome = record.get("expected_outcome")
        if isinstance(expected_outcome, bool):
            response_flag = _parse_boolean(response)
            outcome_pass = response_flag == expected_outcome
            checks.append({"type": "outcome", "passed": outcome_pass, "expected": expected_outcome})
            solved = solved and outcome_pass

        required_keywords = record.get("required_keywords")
        if required_keywords:
            keywords = [str(item) for item in required_keywords] if isinstance(required_keywords, Sequence) else [str(required_keywords)]
            keyword_pass = all(keyword.lower() in response.lower() for keyword in keywords)
            checks.append({"type": "required_keywords", "passed": keyword_pass, "keywords": keywords})
            solved = solved and keyword_pass

        if not checks:
            solved = False
            checks.append({"type": "reference", "passed": False, "reason": "No reference patch or outcome provided."})

        return solved, checks


def _coerce_instance_id(record: Mapping[str, object], fallback_index: int) -> str:
    for key in ("instance_id", "id", "uid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"swe_{fallback_index}"


def _build_prompt(record: Mapping[str, object], instance_id: str) -> str:
    problem_statement = record.get("problem_statement") or record.get("prompt") or record.get("description")
    tests = record.get("tests")
    repo = record.get("repo")
    lines = [
        f"Instance ID: {instance_id}",
        "You are fixing a software bug. Produce a unified diff that resolves the failing tests.",
    ]
    if repo:
        lines.append(f"Repository: {repo}")
    if record.get("base_commit"):
        lines.append(f"Base commit: {record['base_commit']}")
    lines.append("")
    if problem_statement:
        lines.append("Problem Statement:")
        lines.append(str(problem_statement).strip())
        lines.append("")
    if record.get("hints"):
        lines.append("Hints:")
        hints = record["hints"]
        if isinstance(hints, Sequence) and not isinstance(hints, (str, bytes)):
            for hint in hints:
                lines.append(f"- {hint}")
        else:
            lines.append(str(hints))
        lines.append("")
    if tests:
        lines.append("Relevant Tests:")
        if isinstance(tests, Sequence) and not isinstance(tests, (str, bytes)):
            for test in tests:
                lines.append(f"- {test}")
        else:
            lines.append(str(tests))
        lines.append("")
    lines.append("Return only the diff or code modifications needed to fix the issue.")
    lines.append("")
    lines.append("Proposed Patch:")
    return "\n".join(lines)


def _extract_patch(record: Mapping[str, object]) -> str | None:
    for key in ("ground_truth_patch", "expected_patch", "reference_patch", "patch", "diff"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_patch(value)
    return None


def _normalize_patch(text: str) -> str:
    stripped = text.strip()
    lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    return "\n".join(lines)


def _parse_boolean(text: str) -> bool | None:
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "pass", "success", "solved"}:
        return True
    if lowered in {"false", "no", "fail", "failure", "unsolved"}:
        return False
    return None
