from __future__ import annotations

from dataclasses import dataclass
from typing import List

from salience_os_seed.training.microtrainer import MicroTrainer, MicroTrainingJob


@dataclass
class _FakeClock:
    current: float = 0.0

    def now(self) -> float:
        return self.current

    def advance(self, delta: float) -> None:
        self.current += delta


class _DummyModel:
    def __init__(self, clock: _FakeClock, cost: float = 0.05) -> None:
        self.clock = clock
        self.cost = cost
        self.samples: List[str] = []

    def training_step(self, text: str) -> float:
        self.samples.append(text)
        self.clock.advance(self.cost)
        return 0.1


def test_microtrainer_processes_jobs_within_budget() -> None:
    clock = _FakeClock()
    model = _DummyModel(clock, cost=0.05)
    trainer = MicroTrainer(model, default_cpu_budget_s=1.0, time_provider=clock.now)
    job = MicroTrainingJob(identifier="job", samples=["a", "b", "c"])
    trainer.enqueue(job)

    results = trainer.process(cpu_budget_s=0.3)

    assert len(results) == 1
    status = results[0]
    assert status.status == "completed"
    assert status.updates_applied == 3
    assert model.samples == ["a", "b", "c"]


def test_microtrainer_defers_when_cpu_budget_exceeded() -> None:
    clock = _FakeClock()
    model = _DummyModel(clock, cost=0.05)
    trainer = MicroTrainer(model, default_cpu_budget_s=1.0, time_provider=clock.now)
    job = MicroTrainingJob(identifier="job", samples=["a", "b", "c"], cpu_budget_s=0.06)
    trainer.enqueue(job)

    first = trainer.process(cpu_budget_s=1.0)
    assert first and first[0].status == "deferred"
    assert model.samples == ["a", "b"]
    assert trainer.pending() == 1

    second = trainer.process(cpu_budget_s=1.0)
    assert second and second[0].status == "completed"
    assert model.samples == ["a", "b", "c"]
    assert trainer.pending() == 0


def test_microtrainer_cancels_after_repeated_overrun() -> None:
    clock = _FakeClock()
    model = _DummyModel(clock, cost=0.05)
    trainer = MicroTrainer(model, default_cpu_budget_s=1.0, time_provider=clock.now)
    job = MicroTrainingJob(
        identifier="job",
        samples=["a", "b", "c"],
        cpu_budget_s=0.02,
        max_deferrals=1,
    )
    trainer.enqueue(job)

    first = trainer.process(cpu_budget_s=1.0)
    assert first and first[0].status == "deferred"
    second = trainer.process(cpu_budget_s=1.0)
    assert second and second[0].status == "canceled"
    assert trainer.pending() == 0
