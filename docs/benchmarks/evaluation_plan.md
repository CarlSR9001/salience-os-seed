# MONIKA Evaluation Plan

## Overview
This document defines the evaluation harness and protocol for benchmarking MONIKA against `Gemma 3 27B` across seven closed-book benchmarks: `GRIND`, `AIME_2024`, `GPQA`, `SWE_bench`, `MATH_500`, `BFCL`, and `Aider_Polyglot`. All benchmarks carry equal weighting in headline reporting.

## Goals
- Provide reproducible 5-seed runs with 95% confidence intervals.
- Support both local development (≤7B models on RTX 4060/5060-class GPUs) and cloud-scale evaluations (≥13B on A100/H100/GH200).
- Emit machine-readable results suitable for comparisons and model cards.

## Harness Architecture
- Package: `salience_bench`.
- Entry point: `salience-bench` CLI (`python -m salience_bench` for development).
- Configuration:
  - Default manifest `benchmarks/benchmarks_manifest.json`.
  - Per-benchmark adapter modules under `salience_bench/adapters/` (planned structure below).
  - Shared utilities under `salience_bench/utils/`.
- Output:
  - `results/<timestamp>/<benchmark>/<seed>/metrics.json` for raw metrics.
  - Aggregated `results/<timestamp>/summary.json` and Markdown table.

## Benchmark Adapters (planned)
- `GRINDAdapter`
  - Wraps Vellum API. Requires credentials stored in `~/.config/salience_bench/vellum.json` (documented separately).
  - Offline mirror option for prompt replay.
- `AIMEAdapter`
  - Loads from Hugging Face dataset via streaming.
  - Uses short-answer exact-match evaluation.
- `GPQAAdapter`
  - Multiple-choice (A/B/C/D). Evaluate accuracy.
- `SWEbenchAdapter`
  - Supports "Lite" local harness and Verified remote queue.
  - Reports success rate per protocol.
- `MATH500Adapter`
  - Free-form answer with numerical normalization.
- `BFCLAdapter`
  - Function-calling tasks, focusing on match rate and execution success.
- `AiderPolyglotAdapter`
  - Program synthesis tasks; uses Aider reference checker.

Each adapter will implement a common interface defined in `salience_bench/core.py`:
```python
class BenchmarkAdapter(Protocol):
    name: str

    def prepare(self, workdir: Path, seed: int) -> None: ...

    def run(self, model: ModelConfig, seed: int, output_dir: Path) -> MetricSummary: ...

    def aggregate(self, runs: Sequence[MetricSummary]) -> AggregateReport: ...
```

## Seed Handling
- Seeds: `[101, 202, 303, 404, 505]` (default; override via CLI `--seeds`).
- Randomness sources: Python `random`, NumPy, PyTorch, and model-specific sampling must all be seeded.
- Cloud scheduler: include seed in job metadata for traceability.

## Metric Reporting
- Each aggregate report stores mean, standard deviation, and 95% confidence interval.
- Combined leaderboard score: arithmetic mean of normalized benchmark scores (0–100 scale per benchmark definition).
- All results captured in `results/<timestamp>/combined_report.json`.

## Local vs Cloud Execution
- Local (`--backend local`): intended for ≤7B models. Runs sequentially, ensures memory constraints.
- Cloud (`--backend cloud`): produces execution plan JSON (per benchmark + seed) for submission to cluster scheduler (A100/H100/GH200). Harness records job IDs and polls status.
- Shared log format written to `results/<timestamp>/logs/<benchmark>_<seed>.log`.

## Validation
- Include smoke tests under `tests/test_salience_bench.py` to ensure adapters register and CLI generates run plans.
- Continuous integration on GitHub Actions executes lightweight targets:
  - Manifest schema checks.
  - CLI linting (`ruff`/`black`) and unit tests.
  - Optional simulated benchmark stubs for regression testing.

## Documentation Deliverables
- Update `README.md` with quick-start instructions.
- Publish dedicated `docs/benchmarks/<benchmark>_guide.md` for any benchmark requiring manual access (e.g., GRIND credentials).
- Provide `docs/benchmarks/cloud_runbook.md` for scheduling evaluations on remote GPUs.

## Timeline
1. Finalize harness scaffolding and manifests. *(in progress)*
2. Implement adapters iteratively with local fixtures.
3. Integrate with cloud scheduler and record sample runs.
4. Release pip package and GitHub repo alongside documentation.
