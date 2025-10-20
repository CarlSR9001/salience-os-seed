"""Quick baseline runtime probe for manual diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from salience_os_seed.runtime.driver import RuntimeDriver


def main() -> None:
    driver = RuntimeDriver()
    for step in range(5):
        snapshot = driver.step()
        metrics = snapshot.metrics
        decision = metrics.decision
        print(f"step={metrics.step} op={decision.action.operator.name} score={decision.score:.3f}")
        print(
            "  salience:",
            ", ".join(
                f"{key}={value:.3f}" for key, value in sorted(decision.salience_mapping.items())
            ),
        )
        print(
            "  meta:",
            f"verification={metrics.verification_passed} budget={metrics.budget_left:.2f}",
        )
        if metrics.experiment_reports:
            print("  experiments:", metrics.experiment_reports)
        if metrics.maintenance_report:
            print("  maintenance:", metrics.maintenance_report)
    print("done")


if __name__ == "__main__":
    main()
