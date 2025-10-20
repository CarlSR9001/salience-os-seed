from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict as dataclass_asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from . import __version__
from .adapters import REGISTRY
from .core import AggregateReport, BenchmarkAdapter, MetricSummary, ModelConfig

DEFAULT_SEEDS = [101, 202, 303, 404, 505]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Salience benchmark harness")
    parser.add_argument("command", choices=["run", "list"], help="Harness command")
    parser.add_argument(
        "--benchmark",
        "-b",
        action="append",
        help="Benchmark identifier (repeatable). Default: run all registered benchmarks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/benchmarks_manifest.json"),
        help="Path to benchmark manifest JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seed_list",
        type=int,
        help="Custom random seed (repeat). Uses protocol defaults if omitted.",
    )
    parser.add_argument("--model-name", default="monika", help="Model name for reports.")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional revision or checkpoint identifier.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Restrict adapters to workloads runnable on local hardware.",
    )
    parser.add_argument("--print-version", action="store_true", help="Print harness version and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_benchmarks(manifest: dict[str, object]) -> None:
    for name, meta in manifest.items():
        summary = meta.get("notes") or meta.get("description") or ""
        url = meta.get("url", "")
        print(f"- {name}: {summary} {url}".strip())


def resolve_adapters(names: Iterable[str], *, local_only: bool) -> dict[str, BenchmarkAdapter]:
    adapters: dict[str, BenchmarkAdapter] = {}
    for name in names:
        adapter_cls = REGISTRY.get(name)
        if adapter_cls is None:
            options = ", ".join(sorted(REGISTRY))
            raise KeyError(f"Benchmark '{name}' not registered. Available: {options}")
        adapter = adapter_cls()
        requires_cloud = getattr(adapter, "requires_cloud", False)
        if local_only and requires_cloud:
            logging.info("Skipping %s (requires cloud resources)", name)
            continue
        adapters[name] = adapter
    return adapters


def ensure_seeds(seed_list: Sequence[int] | None) -> list[int]:
    if seed_list:
        deduped: dict[int, None] = {}
        for seed in seed_list:
            deduped.setdefault(seed, None)
        return list(deduped.keys())
    return DEFAULT_SEEDS.copy()


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(name=args.model_name, revision=args.model_revision)


def serialize_model_config(cfg: ModelConfig) -> dict[str, object]:
    payload = dataclass_asdict(cfg)
    if payload.get("checkpoint_path") is not None:
        payload["checkpoint_path"] = str(payload["checkpoint_path"])
    return payload


def write_result(path: Path, summary: MetricSummary) -> None:
    payload: dict[str, object] = {
        "benchmark": summary.benchmark,
        "seed": summary.seed,
        "metrics": summary.metrics,
    }
    if summary.metadata:
        payload["metadata"] = summary.metadata
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def aggregate_to_dict(report: AggregateReport) -> dict[str, object]:
    return {
        "benchmark": report.benchmark,
        "mean": report.mean_metrics,
        "std": report.std_metrics,
        "ci95": {metric: list(bounds) for metric, bounds in report.ci95_metrics.items()},
    }


def run_benchmarks(
    selected: Sequence[str] | None,
    manifest: dict[str, object],
    seeds: list[int],
    output: Path,
    model_cfg: ModelConfig,
    *,
    local_only: bool,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    targets = selected or list(manifest.keys())
    adapters = resolve_adapters(targets, local_only=local_only)
    if not adapters:
        raise RuntimeError("No benchmarks selected after applying local/cloud filters.")

    logging.info("Evaluating model '%s' on benchmarks: %s", model_cfg.name, list(adapters))

    plan = {
        "harness_version": __version__,
        "model": serialize_model_config(model_cfg),
        "seeds": seeds,
        "benchmarks": list(adapters),
        "local_only": local_only,
    }
    (run_dir / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    aggregates: list[AggregateReport] = []
    for name, adapter in adapters.items():
        bench_dir = run_dir / name
        bench_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[%s] Preparing assets", name)
        try:
            adapter.prepare(bench_dir, seeds[0])
        except NotImplementedError as exc:
            logging.warning("[%s] prepare() placeholder: %s", name, exc)

        run_results: list[MetricSummary] = []
        for seed in seeds:
            seed_dir = bench_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            logging.info("[%s] Running seed %s", name, seed)
            try:
                result = adapter.run(model_cfg, seed, seed_dir)
            except NotImplementedError as exc:
                logging.warning("[%s] run() placeholder: %s", name, exc)
                result = MetricSummary(
                    benchmark=name,
                    seed=seed,
                    metrics={"status": 0.0},
                    metadata={"note": "adapter not implemented"},
                )
            run_results.append(result)
            write_result(seed_dir / "metrics.json", result)

        logging.info("[%s] Aggregating %d runs", name, len(run_results))
        aggregate = adapter.aggregate(run_results)
        aggregates.append(aggregate)
        with (bench_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(aggregate_to_dict(aggregate), handle, indent=2, sort_keys=True)

    combined = {
        "model": serialize_model_config(model_cfg),
        "generated_at": timestamp,
        "aggregates": [aggregate_to_dict(report) for report in aggregates],
    }
    (run_dir / "combined_report.json").write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Combined report written to %s", run_dir / "combined_report.json")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if args.print_version:
        print(__version__)
        return
    manifest = load_manifest(args.config)
    if args.command == "list":
        list_benchmarks(manifest)
        return
    seeds = ensure_seeds(args.seed_list)
    model_cfg = build_model_config(args)
    if args.command == "run":
        run_benchmarks(
            args.benchmark,
            manifest,
            seeds=seeds,
            output=args.output,
            model_cfg=model_cfg,
            local_only=args.local_only,
        )
        return


if __name__ == "__main__":
    main()
