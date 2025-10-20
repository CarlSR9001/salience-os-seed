"""Quick baseline runtime probe for manual diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _ensure_repo_on_path()
    from salience_os_seed.runtime.driver import RuntimeDriver

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
