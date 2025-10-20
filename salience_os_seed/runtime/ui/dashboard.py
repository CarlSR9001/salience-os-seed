"""Terminal dashboard for SalienceRuntime."""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

try:
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RICH_AVAILABLE = False

from ..driver import DriverSnapshot, RuntimeDriver
from ..orchestrator import RuntimeConfig


@dataclass
class DashboardConfig:
    auto_run: bool = True
    refresh_interval: float = 0.75
    use_rich: Optional[bool] = None


class Dashboard:
    """Simple terminal UI orchestrating runtime interactions."""

    def __init__(self, driver: RuntimeDriver, config: DashboardConfig) -> None:
        self.driver = driver
        self.config = config
        self._running = True
        self._auto = config.auto_run
        self._use_rich = config.use_rich if config.use_rich is not None else RICH_AVAILABLE
        self._console = Console() if self._use_rich else None
        self._lock = threading.Lock()

    def run(self) -> None:
        if self._use_rich:
            self._rich_loop()
        else:
            self._plain_loop()

    def stop(self) -> None:
        self._running = False

    def toggle_auto(self) -> None:
        with self._lock:
            self._auto = not self._auto

    def step_once(self) -> DriverSnapshot:
        with self._lock:
            snapshot = self.driver.step()
            return snapshot

    def _plain_loop(self) -> None:
        while self._running:
            snapshot = self.driver.step()
            self._render_plain(snapshot)
            if not self._auto:
                try:
                    cmd = input("[enter]=step, [a]=auto toggle, [m]=memory verb, [s]=switch gen, [q]=quit: ")
                except EOFError:
                    break
                if not cmd:
                    continue
                self._handle_command(cmd.strip())
            else:
                time.sleep(self.config.refresh_interval)

    def _rich_loop(self) -> None:
        assert self._console is not None
        while self._running:
            snapshot = self.driver.step()
            with self._console.screen():
                self._render_rich(snapshot)
            if not self._auto:
                cmd = self._console.input("[enter]=step, [a]=auto, [m]=memory, [s]=switch, [q]=quit> ")
                if not cmd:
                    continue
                self._handle_command(cmd.strip())
            else:
                time.sleep(self.config.refresh_interval)

    def _handle_command(self, cmd: str) -> None:
        if cmd.lower() == "q":
            self.stop()
            return
        if cmd.lower() == "a":
            self.toggle_auto()
            return
        if cmd.lower() == "m":
            self.driver.inject_memory({"op": "schedule_todo", "text": "user-injected task"})
            return
        if cmd.lower() == "s":
            keys = list(self.driver.available_generators().keys())
            idx = keys.index(self.driver.generator_key)
            next_key = keys[(idx + 1) % len(keys)]
            self.driver.set_generator(next_key)
            return
        # default: single step
        self.driver.step()

    def _render_plain(self, snapshot: DriverSnapshot) -> None:
        columns, _ = shutil.get_terminal_size()
        print("=" * columns)
        print(f"Generator: {snapshot.generator_name} — {snapshot.generator_description}")
        print(snapshot.meta_report)
        print(f"Decision: {snapshot.metrics.decision.action}")
        events = snapshot.metrics.scheduler_snapshot.get("events", [])
        print(f"Scheduler events: {events}")
        print(f"Budget left: {snapshot.metrics.budget_left:.1f}")
        print("Memory todos:")
        for item in snapshot.memory_snapshot.get("todos", []):
            print(f"  - #{item['id']}: {item['text']} (score={item['score']:.2f})")
        scratchpad = getattr(self.driver.runtime, "scratchpad", None)
        if scratchpad is not None:
            print("Scratchpad (current trace):")
            for line in list(scratchpad.current_trace)[-4:]:
                print(f"  - {line}")
            print(f"Scratchpad summary: {scratchpad.summarize(max_traces=3)}")
        maintenance = snapshot.metrics.maintenance_report
        if maintenance:
            print(f"Maintenance: {maintenance}")
        experiments = snapshot.metrics.experiment_reports
        if experiments:
            print("Experiments:")
            for report in experiments:
                print(f"  - {report.get('name', '<unnamed>')}: {report.get('conclusion', '')}")
        print("=" * columns)

    def _render_rich(self, snapshot: DriverSnapshot) -> None:
        assert self._console is not None
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=10),
        )
        layout["body"].split_row(
            Layout(name="decision"),
            Layout(name="status"),
            Layout(name="scratchpad"),
        )

        # Header
        header_panel = Panel(
            Text(snapshot.meta_report, justify="left"),
            title=f"Generator: {snapshot.generator_name}",
            border_style="cyan",
        )
        layout["header"].update(header_panel)

        # Decision details
        decision = snapshot.metrics.decision
        decision_table = Table(show_lines=False)
        decision_table.add_column("Field")
        decision_table.add_column("Value")
        decision_table.add_row("Action", str(decision.action))
        decision_table.add_row("Score", f"{decision.score:.2f}")
        decision_table.add_row("Cooldown", str(decision.cooldown_steps))
        decision_table.add_row("Hysteresis Δ", f"{decision.hysteresis_delta:.2f}")
        layout["decision"].update(Panel(decision_table, title="Decision", border_style="green"))

        # Status & scheduler snapshot
        scheduler = snapshot.metrics.scheduler_snapshot
        events = ", ".join(scheduler.get("events", ())) or "<none>"
        status_table = Table(show_lines=False)
        status_table.add_column("Metric")
        status_table.add_column("Value")
        status_table.add_row("Budget left", f"{snapshot.metrics.budget_left:.1f}")
        status_table.add_row("Idea acceptances", str(snapshot.metrics.idea_acceptances))
        status_table.add_row("Verification", str(snapshot.metrics.verification_passed))
        status_table.add_row("Budget ratio", f"{scheduler.get('budget_ratio', 0.0):.2f}")
        status_table.add_row("Events", events)
        layout["status"].update(Panel(status_table, title="Runtime Status", border_style="yellow"))

        # Scratchpad panel
        scratchpad = getattr(self.driver.runtime, "scratchpad", None)
        if scratchpad is not None:
            current_trace = list(scratchpad.current_trace)
            current_render = "\n".join(current_trace[-8:]) or "<empty>"
            summary = scratchpad.summarize(max_traces=3)
            scratchpad_group = Group(
                Text("Current Trace", style="bold magenta"),
                Text(current_render or "<empty>", overflow="fold"),
                Text("\nRecent Summary", style="bold magenta"),
                Text(summary, overflow="fold"),
            )
        else:
            scratchpad_group = Text("Scratchpad unavailable", style="dim")
        layout["scratchpad"].update(Panel(scratchpad_group, title="Scratchpad", border_style="magenta"))

        # Footer: todos, maintenance, experiments
        todos_table = Table(show_lines=False)
        todos_table.add_column("ID", style="bold")
        todos_table.add_column("Text")
        todos_table.add_column("Score")
        for item in snapshot.memory_snapshot.get("todos", []):
            todos_table.add_row(str(item["id"]), item["text"], f"{item['score']:.2f}")
        if todos_table.row_count == 0:
            todos_table.add_row("-", "<empty>", "0.0")
        todos_panel = Panel(todos_table, title="Todos", border_style="blue")

        maintenance = snapshot.metrics.maintenance_report
        maintenance_text = Text()
        if maintenance:
            for key, value in maintenance.items():
                maintenance_text.append(f"{key}: {value}\n")
        else:
            maintenance_text.append("<none>", style="dim")
        maintenance_panel = Panel(maintenance_text, title="Maintenance", border_style="red")

        experiments = snapshot.metrics.experiment_reports
        experiments_text = Text()
        if experiments:
            for report in experiments:
                name = report.get("name", "<unnamed>")
                conclusion = report.get("conclusion", "")
                experiments_text.append(f"{name}: {conclusion}\n")
        else:
            experiments_text.append("<none>", style="dim")
        experiments_panel = Panel(experiments_text, title="Experiments", border_style="cyan")

        layout["footer"].update(Columns([todos_panel, maintenance_panel, experiments_panel], equal=True, expand=True))

        self._console.print(layout)


def main(auto_run: bool = True, refresh_interval: float = 0.75, use_rich: Optional[bool] = None, generator: str = "baseline") -> None:
    driver = RuntimeDriver(RuntimeConfig())
    if generator in driver.available_generators():
        driver.set_generator(generator)
    config = DashboardConfig(auto_run=auto_run, refresh_interval=refresh_interval, use_rich=use_rich)
    dashboard = Dashboard(driver, config)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        dashboard.stop()


if __name__ == "__main__":
    main()
