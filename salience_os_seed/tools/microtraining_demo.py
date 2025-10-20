"""CLI utility to exercise the micro-training loop with telemetry output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from ..proto_lm.trainer import ProtoLanguageModel, TrainingConfig
from ..telemetry import BUS, TelemetryEvent, render_parameter_event, render_training_event
from ..training.microtrainer import JobStatus, MicroTrainer, MicroTrainingJob


DEFAULT_SAMPLES: Sequence[str] = (
    "Micro-training keeps the proto LM responsive to new context.",
    "Each job processes just a handful of sequences inside a safe budget.",
    "Telemetry should stream updates so we can monitor the loop.",
)


def _load_samples(file: Path | None, *, repeat: int) -> list[str]:
    samples: List[str] = []
    if file is not None:
        try:
            data = file.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - CLI level validation
            raise SystemExit(f"Failed to read sample file '{file}': {exc}") from exc
        samples = [line.strip() for line in data.splitlines() if line.strip()]
    if not samples:
        samples = list(DEFAULT_SAMPLES)
    repeat = max(1, int(repeat))
    return samples * repeat


def _telemetry_sink(event: TelemetryEvent) -> None:
    if event.type == "training/step":
        for line in render_training_event(event):
            print(line)
    elif event.type == "parameters/update":
        print(render_parameter_event(event))


def _process_jobs(trainer: MicroTrainer, *, cpu_budget: float, max_cycles: int) -> list[JobStatus]:
    cycles = max(1, max_cycles)
    results: list[JobStatus] = []
    for _ in range(cycles):
        if trainer.pending() == 0:
            break
        batch = trainer.process(cpu_budget_s=cpu_budget)
        if not batch:
            print("No jobs were processed within the CPU budget; consider increasing it.")
            break
        for status in batch:
            print(
                "[job] {identifier} status={status} updates={updates} elapsed={elapsed:.3f}s reason={reason}".format(
                    identifier=status.identifier,
                    status=status.status,
                    updates=status.updates_applied,
                    elapsed=status.elapsed_s,
                    reason=status.reason or "-",
                )
            )
        results.extend(batch)
        if all(status.status != "deferred" for status in batch):
            continue
    else:
        if trainer.pending() > 0:
            print("Reached maximum processing cycles while jobs remain queued.")
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a micro-training demo job with telemetry streaming.")
    parser.add_argument("--samples", type=Path, default=None, help="Optional newline-delimited text file to train on.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the provided samples this many times.")
    parser.add_argument("--cpu-budget", type=float, default=0.25, help="CPU budget (seconds) per processing cycle.")
    parser.add_argument("--job-budget", type=float, default=None, help="Optional per-job CPU budget override.")
    parser.add_argument("--max-updates", type=int, default=None, help="Limit the number of updates per job.")
    parser.add_argument("--max-cycles", type=int, default=8, help="Maximum number of processing cycles to run.")
    parser.add_argument(
        "--no-telemetry", action="store_true", help="Disable printing telemetry events to stdout."
    )
    parser.add_argument("--identifier", type=str, default="microtraining-demo", help="Identifier to tag the job with.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    samples = _load_samples(args.samples, repeat=args.repeat)
    if not samples:
        raise SystemExit("No training samples were provided.")

    cpu_budget = max(0.0, float(args.cpu_budget))
    model = ProtoLanguageModel(TrainingConfig())
    trainer = MicroTrainer(model, default_cpu_budget_s=cpu_budget)
    job = MicroTrainingJob(
        identifier=args.identifier,
        samples=samples,
        metadata={"sample_count": len(samples)},
        max_updates=args.max_updates,
        cpu_budget_s=args.job_budget,
    )
    trainer.enqueue(job)

    unsubscribe = BUS.subscribe(_telemetry_sink) if not args.no_telemetry else None
    try:
        statuses = _process_jobs(trainer, cpu_budget=cpu_budget, max_cycles=args.max_cycles)
    finally:
        if unsubscribe is not None:
            unsubscribe()

    if trainer.pending() > 0:
        print(f"{trainer.pending()} job(s) are still pending after the demo run.")

    total_updates = sum(status.updates_applied for status in statuses)
    print(
        f"Completed {total_updates} updates across {len(statuses)} cycle(s); model step is now {model.step}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

