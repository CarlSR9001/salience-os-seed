from __future__ import annotations

from typing import List

from salience_os_seed.proto_lm.trainer import ProtoLanguageModel, TrainingConfig
from salience_os_seed.telemetry import BUS, TelemetryEvent
from salience_os_seed.training.microtrainer import MicroTrainer, MicroTrainingJob


class _FakeClock:
    current: float = 0.0

    def now(self) -> float:
        return self.current


def test_microtrainer_emits_training_telemetry() -> None:
    clock = _FakeClock()
    model = ProtoLanguageModel(TrainingConfig())
    trainer = MicroTrainer(model, default_cpu_budget_s=1.0, time_provider=clock.now)
    job = MicroTrainingJob(identifier="telemetry", samples=["alpha", "beta", "gamma"])
    trainer.enqueue(job)

    events: List[TelemetryEvent] = []
    unsubscribe = BUS.subscribe(events.append)
    try:
        results = trainer.process(cpu_budget_s=1.0)
    finally:
        unsubscribe()

    assert results and results[0].status == "completed"
    assert model.step == len(job.samples)

    training_events = [event for event in events if event.type == "training/step"]
    assert len(training_events) == len(job.samples)
    for index, event in enumerate(training_events, start=1):
        payload = event.payload
        assert payload.get("step") == index
        loss = payload.get("loss")
        assert isinstance(loss, float)
        assert payload.get("parameter_total")
        grads = payload.get("grads")
        assert grads
