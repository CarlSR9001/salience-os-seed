import math

import pytest

from salience_os_seed.runtime.driver import RuntimeDriver
from salience_os_seed.runtime.state_gen import BaselineGenerator, create_default_generators


def test_baseline_generator_reset_restarts_counter():
    generator = BaselineGenerator()
    first_state = generator.next_state()
    generator.next_state()  # advance
    generator.reset()
    reset_state = generator.next_state()
    assert math.isclose(
        first_state["prediction"]["steps_remaining"],
        reset_state["prediction"]["steps_remaining"],
        rel_tol=1e-6,
    )
    assert reset_state["context"]["tokens"][0] == "baseline"


def test_driver_step_records_history_and_memory():
    driver = RuntimeDriver()
    driver.set_generator("baseline")
    snapshot1 = driver.step()
    assert snapshot1.generator_name == "baseline"
    assert len(snapshot1.last_metrics) == 1
    assert snapshot1.metrics.step == 1

    driver.inject_memory({"op": "schedule_todo", "text": "write tests"})
    snapshot2 = driver.step()
    todos = snapshot2.memory_snapshot.get("todos", [])
    assert any(entry["text"] == "write tests" for entry in todos)
    assert len(snapshot2.last_metrics) == 2


def test_driver_switches_generators():
    driver = RuntimeDriver()
    keys = list(create_default_generators().keys())
    assert len(keys) >= 1
    driver.set_generator(keys[0])
    snapshot = driver.step()
    assert snapshot.generator_name == keys[0]
    if len(keys) > 1:
        driver.set_generator(keys[1])
        snapshot2 = driver.step()
        assert snapshot2.generator_name == keys[1]
    available = driver.available_generators()
    for key in keys:
        assert key in available
        assert isinstance(available[key], str)
