"""CLI entry point for the SalienceRuntime dashboard."""

from __future__ import annotations

import argparse
import time

from ..driver import RuntimeDriver
from ..orchestrator import RuntimeConfig
from ..state_gen import create_default_generators
from .dashboard import Dashboard, DashboardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SalienceRuntime interactive console")
    default_generators = create_default_generators()
    parser.add_argument("--generator", choices=list(default_generators.keys()), default="baseline")
    parser.add_argument("--auto", action="store_true", help="Enable auto-run loop on start")
    parser.add_argument("--interval", type=float, default=0.75, help="Auto refresh interval")
    parser.add_argument("--rich", action="store_true", help="Force rich rendering (if installed)")
    parser.add_argument("--plain", action="store_true", help="Force plain rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    driver = RuntimeDriver(RuntimeConfig())
    driver.set_generator(args.generator)
    config = DashboardConfig(
        auto_run=args.auto,
        refresh_interval=args.interval,
        use_rich=True if args.rich else False if args.plain else None,
    )
    dashboard = Dashboard(driver, config)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        dashboard.stop()
    finally:
        time.sleep(0.2)


if __name__ == "__main__":
    main()
