import ast
import importlib.util
from pathlib import Path

import pytest


def _load_experiments_module():
    module_path = Path(__file__).resolve().parents[1] / "salience_os_seed" / "core" / "ideas" / "experiments.py"
    spec = importlib.util.spec_from_file_location("salience_os_seed.core.ideas.experiments", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ExperimentDispatcher = _load_experiments_module().ExperimentDispatcher


def _runtime_default_parameters() -> tuple[str, ...]:
    config_path = Path(__file__).resolve().parents[1] / "salience_os_seed" / "runtime" / "config.py"
    module = ast.parse(config_path.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "ExperimentConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", None) == "parameters":
                    call = stmt.value
                    if not isinstance(call, ast.Call):
                        continue
                    for keyword in call.keywords:
                        if keyword.arg == "default_factory" and isinstance(keyword.value, ast.Lambda):
                            return tuple(ast.literal_eval(keyword.value.body))
    raise AssertionError("Failed to locate runtime experiment default parameters")


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


def test_experiment_dispatcher_maps_scheduler_aliases():
    dispatcher = ExperimentDispatcher(["scheduler.threshold"], max_concurrent=1)
    experiment = dispatcher.propose_experiment({"uncertainty": 0.7, "drag": 0.2}, False)
    assert experiment is not None
    assert "scheduler.min_budget_ratio" in experiment.parameter_overrides
    assert experiment.parameter_metadata["scheduler.threshold"] == "scheduler.min_budget_ratio"


def test_experiment_dispatcher_uses_runtime_defaults():
    dispatcher = ExperimentDispatcher(_runtime_default_parameters(), max_concurrent=1)
    experiment = dispatcher.propose_experiment({"uncertainty": 0.8, "drag": 0.3}, False)
    assert experiment is not None
    assert set(experiment.parameter_overrides.keys()) == {
        "controller.lambda_cost",
        "scheduler.min_budget_ratio",
    }
