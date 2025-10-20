"""Micro-training job coordinator for MCP scheduled updates."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable, Mapping, Optional, Sequence

from ..proto_lm.trainer import ProtoLanguageModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MicroTrainingJob:
    """Container describing a batch of text updates to run incrementally."""

    identifier: str
    samples: Sequence[str]
    metadata: Mapping[str, object] = field(default_factory=dict)
    max_updates: Optional[int] = None
    cpu_budget_s: Optional[float] = None
    max_deferrals: int = 3

    _cursor: int = field(init=False, default=0)
    _updates: int = field(init=False, default=0)
    _deferrals: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.samples = tuple(str(sample) for sample in self.samples)

    def next_sample(self) -> Optional[str]:
        """Return the next training sample if available."""

        if self.max_updates is not None and self._updates >= self.max_updates:
            return None
        if self._cursor >= len(self.samples):
            return None
        sample = self.samples[self._cursor]
        self._cursor += 1
        self._updates += 1
        return sample

    def has_work(self) -> bool:
        """Determine whether additional updates are pending for this job."""

        if self.max_updates is not None and self._updates >= self.max_updates:
            return False
        return self._cursor < len(self.samples)

    @property
    def updates_completed(self) -> int:
        return self._updates

    def mark_deferred(self) -> bool:
        """Increment the deferral counter and report whether the job can resume."""

        self._deferrals += 1
        return self._deferrals <= self.max_deferrals


@dataclass(frozen=True)
class JobStatus:
    """Outcome summary returned after processing queued jobs."""

    identifier: str
    status: str
    updates_applied: int
    elapsed_s: float
    metadata: Mapping[str, object]
    reason: Optional[str] = None


class MicroTrainer:
    """Consume MCP-scheduled micro-training jobs under CPU budgets."""

    def __init__(
        self,
        model: ProtoLanguageModel,
        *,
        default_cpu_budget_s: float = 0.25,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self.model = model
        self.default_cpu_budget_s = max(0.0, float(default_cpu_budget_s))
        self._queue: Deque[MicroTrainingJob] = deque()
        self._lock = threading.Lock()
        self._time = time_provider or time.perf_counter

    def enqueue(self, job: MicroTrainingJob) -> None:
        """Add a job to the processing queue."""

        with self._lock:
            self._queue.append(job)

    def extend(self, jobs: Iterable[MicroTrainingJob]) -> None:
        """Append multiple jobs to the queue."""

        with self._lock:
            self._queue.extend(jobs)

    def pending(self) -> int:
        """Return the number of queued jobs."""

        with self._lock:
            return len(self._queue)

    def _pop_job(self) -> Optional[MicroTrainingJob]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def _requeue(self, job: MicroTrainingJob) -> None:
        with self._lock:
            self._queue.append(job)

    def process(self, *, cpu_budget_s: Optional[float] = None) -> list[JobStatus]:
        """Process queued jobs within the provided CPU budget."""

        results: list[JobStatus] = []
        total_budget = self.default_cpu_budget_s if cpu_budget_s is None else max(0.0, float(cpu_budget_s))
        deadline = self._time() + total_budget if total_budget > 0 else self._time()

        while True:
            now = self._time()
            if total_budget > 0 and now >= deadline:
                break
            job = self._pop_job()
            if job is None:
                break
            job_start = self._time()
            job_budget = job.cpu_budget_s if job.cpu_budget_s is not None else total_budget
            job_deadline = job_start + max(0.0, job_budget)
            updates_before = job.updates_completed
            status = "completed"
            reason: Optional[str] = None
            error: Optional[Exception] = None

            try:
                while True:
                    current_time = self._time()
                    if total_budget > 0 and current_time >= deadline:
                        status = "deferred"
                        reason = "global-budget-exhausted"
                        break
                    if job_budget > 0 and current_time >= job_deadline:
                        status = "deferred"
                        reason = "cpu-budget-exceeded"
                        break
                    sample = job.next_sample()
                    if sample is None:
                        break
                    self.model.training_step(sample)
                    if not job.has_work():
                        break
            except Exception as exc:  # pragma: no cover - defensive safety
                logger.exception("Error executing micro-training job %s", job.identifier)
                status = "error"
                reason = str(exc)
                error = exc

            elapsed = self._time() - job_start
            updates_applied = job.updates_completed - updates_before

            if error is not None:
                results.append(
                    JobStatus(
                        identifier=job.identifier,
                        status="error",
                        updates_applied=updates_applied,
                        elapsed_s=elapsed,
                        metadata=job.metadata,
                        reason=reason,
                    )
                )
                continue

            if job.has_work():
                if reason is None:
                    reason = "work-remaining"
                    status = "deferred"
                if reason == "cpu-budget-exceeded" and not job.mark_deferred():
                    status = "canceled"
                    reason = "cpu-budget-overrun"
                if status == "deferred":
                    self._requeue(job)
                elif status == "canceled":
                    logger.warning(
                        "Canceling MCP job %s after exceeding CPU budget %s times", job.identifier, job.max_deferrals
                    )
            else:
                status = "completed"

            results.append(
                JobStatus(
                    identifier=job.identifier,
                    status=status,
                    updates_applied=updates_applied,
                    elapsed_s=elapsed,
                    metadata=job.metadata,
                    reason=reason,
                )
            )

            if status == "deferred" and reason == "cpu-budget-exceeded":
                break
            if total_budget > 0 and self._time() >= deadline:
                break

        return results


__all__ = ["MicroTrainer", "MicroTrainingJob", "JobStatus"]
