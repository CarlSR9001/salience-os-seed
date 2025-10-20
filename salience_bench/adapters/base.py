from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..core import AggregateReport, BenchmarkAdapter, MetricSummary, ModelConfig, SimpleAggregateAdapter

logger = logging.getLogger(__name__)


class CompletionFn(Protocol):
    """Callable used by adapters to obtain model completions."""

    def __call__(
        self,
        prompt: str,
        *,
        model: ModelConfig,
        seed: int,
        metadata: Mapping[str, object] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class AdapterRuntimeConfig:
    """Runtime configuration shared by dataset-backed adapters."""

    cache_root: Path
    hf_cache_dir: Path
    hf_token: str | None
    completion_fn_path: str | None
    dataset_overrides: dict[str, Path]
    options: dict[str, dict[str, str]]

    @classmethod
    def from_env(cls) -> "AdapterRuntimeConfig":
        cache_root = Path(os.getenv("SALIENCE_BENCH_CACHE_DIR", Path.home() / ".cache" / "salience_bench")).expanduser()
        hf_cache = Path(
            os.getenv("SALIENCE_BENCH_HF_CACHE_DIR", os.getenv("HF_DATASETS_CACHE", str(cache_root / "hf")))
        ).expanduser()
        hf_token = (
            os.getenv("SALIENCE_HF_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        completion_path = os.getenv("SALIENCE_BENCH_COMPLETION_FN")

        dataset_overrides: dict[str, Path] = {}
        prefix = "SALIENCE_BENCH_DATA_"
        for key, value in os.environ.items():
            if key.startswith(prefix) and value:
                dataset_overrides[_normalize_key(key[len(prefix) :])] = Path(value).expanduser()

        options: dict[str, dict[str, str]] = {}
        opt_prefix = "SALIENCE_BENCH_OPTION_"
        for key, value in os.environ.items():
            if key.startswith(opt_prefix) and value:
                suffix = key[len(opt_prefix) :]
                parts = suffix.split("_", 1)
                if len(parts) != 2:
                    continue
                adapter_key = _normalize_key(parts[0])
                option_key = parts[1].lower()
                options.setdefault(adapter_key, {})[option_key] = value

        return cls(
            cache_root=cache_root,
            hf_cache_dir=hf_cache,
            hf_token=hf_token,
            completion_fn_path=completion_path,
            dataset_overrides=dataset_overrides,
            options=options,
        )

    def dataset_override(self, benchmark: str) -> Path | None:
        return self.dataset_overrides.get(_normalize_key(benchmark))

    def dataset_cache_dir(self, benchmark: str) -> Path:
        target = self.cache_root / _normalize_key(benchmark)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_option(self, benchmark: str, key: str, default: str | None = None) -> str | None:
        return self.options.get(_normalize_key(benchmark), {}).get(key.lower(), default)


def _normalize_key(raw: str) -> str:
    return raw.strip().lower().replace("/", "_").replace("-", "_")


def import_from_string(target: str) -> object:
    """Resolve ``module:attribute`` or ``module.attribute`` references."""

    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def resolve_completion_fn(config: AdapterRuntimeConfig) -> CompletionFn:
    if config.completion_fn_path is None:
        raise RuntimeError(
            "No completion callback configured. Set SALIENCE_BENCH_COMPLETION_FN or pass a completion_fn explicitly."
        )
    fn = import_from_string(config.completion_fn_path)
    if not callable(fn):  # pragma: no cover - defensive guard
        raise TypeError(f"Resolved completion callable '{config.completion_fn_path}' is not callable")
    return fn  # type: ignore[return-value]


def load_local_records(path: Path) -> list[dict[str, object]]:
    """Load benchmark records from a JSON or JSONL payload."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset override path not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, object]] = []
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
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [dict(item) for item in payload["data"]]
        return [dict(payload)]
    raise ValueError(f"Unsupported dataset override structure in {path}: {type(payload)!r}")


class UnimplementedAdapter(SimpleAggregateAdapter):
    """Adapter skeleton that marks TODO implementations."""

    def __init__(self, name: str) -> None:
        self.name = name

    def prepare(self, workdir: Path, seed: int) -> None:  # noqa: D401 - simple implementation
        logger.warning("prepare() for %s not yet implemented", self.name)

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary:
        raise NotImplementedError(f"Benchmark adapter '{self.name}' run() not implemented yet")


def aggregate_placeholder(name: str, runs: Sequence[MetricSummary]) -> AggregateReport:
    adapter = UnimplementedAdapter(name)
    return adapter.aggregate(runs)
