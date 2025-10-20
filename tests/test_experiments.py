import pytest

from salience_os_seed.core.ideas import ExperimentDispatcher


def test_experiment_dispatcher_proposes_and_analyzes():
    dispatcher = ExperimentDispatcher(["controller.lambda_cost"], max_concurrent=1)
    salience = {"uncertainty": 0.8, "drag": 0.2}
    experiment = dispatcher.propose_experiment(salience, verification_success=False)
    assert experiment is not None
    assert experiment.parameter_overrides

    def step_fn(overrides):
        assert "controller.lambda_cost" in overrides
        return {"verification": 0.9, "cost": 0.3}

    dispatcher.run_experiment(experiment, step_fn)
    experiment.baseline_verification = 0.2
    analysed = dispatcher.analyze_results(experiment)
    assert analysed.conclusion == "positive"
    assert analysed.results["verification_mean"] >= 0.9


def test_experiment_dispatcher_respects_max_concurrent():
    dispatcher = ExperimentDispatcher(["controller.lambda_cost"], max_concurrent=1)
    salience = {"uncertainty": 0.9, "drag": 0.1}
    first = dispatcher.propose_experiment(salience, verification_success=False)
    assert first is not None
    second = dispatcher.propose_experiment(salience, verification_success=False)
    assert second is None
    dispatcher.active.clear()
    second = dispatcher.propose_experiment(salience, verification_success=False)
    assert second is not None
