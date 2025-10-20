from salience_os_seed.core.scheduler import EventDrivenScheduler
from salience_os_seed.core.scheduler.events import default_registry


def test_scheduler_fires_on_uncertainty_spike():
    scheduler = EventDrivenScheduler(registry=default_registry())
    salience = {
        "uncertainty": 2.5,
        "novelty": 0.3,
        "progress": 0.4,
        "drag": 0.2,
        "cost": 0.1,
    }

    decision_operator = "SASS"
    assert scheduler.should_fire(salience, decision_operator, budget_left=100.0, budget_total=200.0)
    snapshot = scheduler.snapshot()
    assert "uncertainty_spike" in snapshot["events"]


def test_scheduler_respects_budget_floor():
    scheduler = EventDrivenScheduler()
    salience = {
        "uncertainty": 2.0,
        "novelty": 0.0,
        "progress": 0.0,
        "drag": 0.0,
        "cost": 0.0,
    }
    # Budget near zero should allow only VERIFY
    assert scheduler.should_fire(salience, "VERIFY", budget_left=1.0, budget_total=100.0)
    assert not scheduler.should_fire(salience, "SASS", budget_left=1.0, budget_total=100.0)
