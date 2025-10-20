"""Runtime smoke test wiring new Tier-2 subsystems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import RuntimeConfig
from .orchestrator import SalienceRuntime


DEFAULT_STATE = {
    "context": {
        "tokens": ["smoke", "test", "input"],
        "text": "smoke test input",
    },
    "prediction": {
        "token_logits": [[0.1, 0.2, 0.3]],
        "entropy_estimate": 0.5,
        "steps_remaining": 1.0,
    },
    "decision_proposal": {"operator": "VERIFY", "cot_depth": 1},
    "token_cost": 10,
}


def run_smoke(times: int = 3, config: Optional[RuntimeConfig] = None, output: Optional[str] = None) -> None:
    runtime = SalienceRuntime(config)
    metrics_log = []
    for _ in range(times):
        metrics = runtime.run_step(DEFAULT_STATE)
        metrics_log.append(
            {
                "step": metrics.step,
                "maintenance": metrics.maintenance_report,
                "experiments": list(metrics.experiment_reports),
                "episode": metrics.episode_recorded,
                "verification": metrics.verification_passed,
            }
        )
    if output:
        Path(output).write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")
    else:
        print(json.dumps(metrics_log, indent=2))


if __name__ == "__main__":
    run_smoke()
