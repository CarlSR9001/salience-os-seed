"""Terminal dashboard for SalienceRuntime with enhanced UX."""

from __future__ import annotations

from collections import deque
import itertools
import shutil
import threading
import time
import textwrap
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

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

from ..config import RuntimeConfig
from ..driver import DriverSnapshot, RuntimeDriver


@dataclass
class DashboardConfig:
    auto_run: bool = True
    refresh_interval: float = 0.75
    use_rich: Optional[bool] = None


_SPARKLINE_BARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float], *, width: int = 16) -> str:
    if not values:
        return "<no data>"
    tail = list(values)[-width:]
    minimum = min(tail)
    maximum = max(tail)
    if maximum == minimum:
        return _SPARKLINE_BARS[0] * len(tail)
    scale = len(_SPARKLINE_BARS) - 1
    normalized = [int((value - minimum) / (maximum - minimum) * scale) for value in tail]
    return "".join(_SPARKLINE_BARS[index] for index in normalized)


def _format_plain_header(text: str, *, width: int) -> str:
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent="    ")
    return "\n".join(wrapper.fill(line) for line in text.splitlines())


def _format_plain_table(rows: Sequence[Sequence[str]], *, headers: Sequence[str], width: int) -> str:
    if not rows:
        empty_row = tuple("—" if index == 0 else "" for index in range(len(headers)))
        rows = [empty_row]
    col_widths = [
        max(len(str(cell)) for cell in itertools.chain([header], (row[i] for row in rows)))
        for i, header in enumerate(headers)
    ]
    col_widths = [min(w, max(8, width // max(len(headers), 1) - 2)) for w in col_widths]

    def render_row(row: Sequence[str]) -> str:
        padded = [str(cell)[: col_widths[i]].ljust(col_widths[i]) for i, cell in enumerate(row)]
        return " | ".join(padded)

    header_line = render_row(headers)
    separator = "-" * len(header_line)
    body = "\n".join(render_row(tuple(row)) for row in rows)
    return f"{header_line}\n{separator}\n{body}"


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
        self._messages: Deque[str] = deque(maxlen=8)
        self._post_message("Dashboard ready. Press 'h' for help.")

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
        snapshot = self.driver.snapshot()
        while self._running:
            self._render_plain(snapshot)
            if self._auto:
                time.sleep(self.config.refresh_interval)
                snapshot = self.step_once()
                continue
            try:
                raw = input(self._command_prompt())
            except EOFError:
                break
            snapshot = self._process_command(raw)

    def _rich_loop(self) -> None:
        assert self._console is not None
        snapshot = self.driver.snapshot()
        while self._running:
            with self._console.screen():
                self._render_rich(snapshot)
            if self._auto:
                time.sleep(self.config.refresh_interval)
                snapshot = self.step_once()
                continue
            cmd = self._console.input(self._command_prompt())
            snapshot = self._process_command(cmd)

    def _process_command(self, raw: str) -> DriverSnapshot:
        cmd = raw.strip()
        if not cmd:
            return self.step_once()
        normalized = cmd.lower()
        if normalized in {"q", "quit", "exit"}:
            self.stop()
            self._post_message("Shutting down dashboard.")
            return self.driver.snapshot()
        if normalized in {"a", "auto"}:
            self.toggle_auto()
            state = "enabled" if self._auto else "paused"
            self._post_message(f"Auto-run {state}.")
            return self.driver.snapshot()
        if normalized in {"m", "memory"}:
            self.driver.inject_memory({"op": "schedule_todo", "text": "user-injected task"})
            self._post_message("Injected todo into structured memory.")
            return self.driver.snapshot()
        if normalized in {"s", "switch"}:
            keys = list(self.driver.available_generators().keys())
            idx = keys.index(self.driver.generator_key)
            next_key = keys[(idx + 1) % len(keys)]
            self.driver.set_generator(next_key)
            self._post_message(f"Generator switched to '{next_key}'.")
            return self.driver.snapshot()
        if normalized in {"r", "reset"}:
            self.driver.reset()
            self._post_message("Runtime reset.")
            return self.driver.snapshot()
        if normalized in {"h", "?", "help"}:
            self._post_message("Commands: " + ", ".join(self._command_descriptions()))
            return self.driver.snapshot()
        if normalized.startswith("g "):
            target = normalized.split(maxsplit=1)[1]
            try:
                self.driver.set_generator(target)
            except KeyError:
                self._post_message(f"Unknown generator '{target}'.")
            else:
                self._post_message(f"Generator switched to '{target}'.")
            return self.driver.snapshot()
        # default: single step
        return self.step_once()

    def _render_plain(self, snapshot: DriverSnapshot) -> None:
        columns, _ = shutil.get_terminal_size(fallback=(120, 40))
        divider = "=" * columns
        print(divider)
        auto_state = "ON" if self._auto else "OFF"
        header = f"Step {snapshot.metrics.step} • Auto {auto_state} • Generator {snapshot.generator_name}"
        print(header)
        print(_format_plain_header(snapshot.generator_description, width=columns))
        print(_format_plain_header(snapshot.meta_report, width=columns))
        print("-" * columns)

        decision_rows = [
            ("Action", str(snapshot.metrics.decision.action)),
            ("Score", f"{snapshot.metrics.decision.score:.2f}"),
            ("Cooldown", str(snapshot.metrics.decision.cooldown_steps)),
            ("Hysteresis Δ", f"{snapshot.metrics.decision.hysteresis_delta:.2f}"),
        ]
        decision_table = _format_plain_table(
            [[key, value] for key, value in decision_rows], headers=("Field", "Value"), width=columns
        )
        print("Decision")
        print(decision_table)

        scheduler = snapshot.metrics.scheduler_snapshot
        events = ", ".join(scheduler.get("events", ())) or "<none>"
        status_rows = [
            ("Budget left", f"{snapshot.metrics.budget_left:.1f}"),
            ("Budget ratio", f"{scheduler.get('budget_ratio', 0.0):.2f}"),
            ("Idea acceptances", str(snapshot.metrics.idea_acceptances)),
            ("Verification", str(snapshot.metrics.verification_passed)),
            ("Events", events),
        ]
        status_table = _format_plain_table(
            [[key, value] for key, value in status_rows], headers=("Metric", "Value"), width=columns
        )
        print("Status")
        print(status_table)

        history_lines = []
        for metrics in list(snapshot.last_metrics)[-8:][::-1]:
            history_lines.append(
                f"#{metrics.step:04d} {str(metrics.decision.action):20} budget={metrics.budget_left:6.1f} verify={metrics.verification_passed}"
            )
        if history_lines:
            print("Recent Steps")
            for line in history_lines:
                print(f"  {line}")

        salience_items = sorted(
            snapshot.metrics.salience_raw.items(), key=lambda item: -abs(item[1])
        )[:6]
        if salience_items:
            salience_rows = [(key, f"{value:.2f}") for key, value in salience_items]
            salience_table = _format_plain_table(salience_rows, headers=("Channel", "Value"), width=columns)
            print("Salience")
            print(salience_table)

        todos = [
            (str(item["id"]), item["text"], f"{item['score']:.2f}")
            for item in snapshot.memory_snapshot.get("todos", [])
        ]
        todos_table = _format_plain_table(todos, headers=("ID", "Text", "Score"), width=columns)
        print("Todos")
        print(todos_table)

        scratchpad = getattr(self.driver.runtime, "scratchpad", None)
        if scratchpad is not None:
            print("Scratchpad")
            current_trace = list(scratchpad.current_trace)[-6:]
            for line in current_trace:
                print(f"  {line}")
            summary = scratchpad.summarize(max_traces=3)
            print(_format_plain_header(f"Summary: {summary}", width=columns))

        maintenance = snapshot.metrics.maintenance_report
        if maintenance:
            print("Maintenance")
            for key, value in maintenance.items():
                print(f"  {key}: {value}")

        experiments = snapshot.metrics.experiment_reports
        if experiments:
            print("Experiments")
            for report in experiments:
                name = report.get("name", "<unnamed>")
                conclusion = report.get("conclusion", "")
                print(f"  {name}: {conclusion}")

        print("Commands:")
        print("  " + "; ".join(self._command_descriptions()))

        generators = []
        for key, description in self.driver.available_generators().items():
            prefix = "*" if key == snapshot.generator_name else " "
            generators.append(f"{prefix} {key}: {description}")
        print("Generators:")
        for line in generators:
            print(f"  {line}")

        if self._messages:
            print("Messages:")
            for message in self._messages:
                print(f"  - {message}")

        print(divider)

    def _render_rich(self, snapshot: DriverSnapshot) -> None:
        assert self._console is not None
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=14),
        )
        layout["body"].split_row(
            Layout(name="decision"),
            Layout(name="status"),
            Layout(name="scratchpad"),
        )

        layout["footer"].split_column(
            Layout(name="footer_top"),
            Layout(name="footer_bottom", size=5),
        )

        # Header
        auto_state = "ON" if self._auto else "OFF"
        header_lines = Text()
        header_lines.append(f"Auto: {auto_state} • Generator: {snapshot.generator_name}\n", style="bold cyan")
        header_lines.append(snapshot.generator_description + "\n", style="dim")
        header_lines.append(snapshot.meta_report)
        header_panel = Panel(
            header_lines,
            title=f"Step {snapshot.metrics.step}",
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
        history = list(snapshot.last_metrics)
        budget_trend = _sparkline([metric.budget_left for metric in history])
        verification_trend = _sparkline([
            1.0 if metric.verification_passed else 0.0 for metric in history
        ])
        trend_table = Table(show_lines=False)
        trend_table.add_column("Signal")
        trend_table.add_column("Trend")
        trend_table.add_row("Budget", budget_trend)
        trend_table.add_row("Verification", verification_trend)
        status_group = Group(status_table, trend_table)
        layout["status"].update(Panel(status_group, title="Runtime Status", border_style="yellow"))

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

        layout["footer_top"].update(
            Columns([todos_panel, maintenance_panel, experiments_panel], equal=True, expand=True)
        )

        salience_table = Table(show_header=True, header_style="bold")
        salience_table.add_column("Channel")
        salience_table.add_column("Value", justify="right")
        if snapshot.metrics.salience_raw:
            for key, value in sorted(
                snapshot.metrics.salience_raw.items(), key=lambda item: -abs(item[1])
            )[:6]:
                salience_table.add_row(key, f"{value:.2f}")
        else:
            salience_table.add_row("<none>", "0.00")
        salience_panel = Panel(salience_table, title="Salience", border_style="blue")

        history_table = Table(show_header=True, header_style="bold")
        history_table.add_column("Step", justify="right")
        history_table.add_column("Action")
        history_table.add_column("Budget", justify="right")
        history_table.add_column("Verify", justify="right")
        for metrics in history[-6:][::-1]:
            history_table.add_row(
                str(metrics.step),
                str(metrics.decision.action),
                f"{metrics.budget_left:.1f}",
                "✓" if metrics.verification_passed else "✗",
            )
        history_panel = Panel(history_table, title="Recent Steps", border_style="white")

        commands_text = Text()
        for description in self._command_descriptions():
            commands_text.append(f"{description}\n")
        commands_panel = Panel(commands_text, title="Commands", border_style="cyan")

        generator_table = Table(show_header=True, header_style="bold")
        generator_table.add_column("Key")
        generator_table.add_column("Description")
        for key, description in self.driver.available_generators().items():
            style = "bold green" if key == snapshot.generator_name else ""
            generator_table.add_row(key, description, style=style)
        generator_panel = Panel(generator_table, title="Generators", border_style="green")

        messages_text = Text()
        if self._messages:
            for message in self._messages:
                messages_text.append(f"• {message}\n")
        else:
            messages_text.append("<none>", style="dim")
        messages_panel = Panel(messages_text, title="Messages", border_style="magenta")

        footer_bottom_group = Columns(
            [salience_panel, history_panel, commands_panel, generator_panel, messages_panel],
            expand=True,
        )
        layout["footer_bottom"].update(footer_bottom_group)

        self._console.print(layout)

    def _post_message(self, message: str) -> None:
        self._messages.appendleft(message)

    def _command_descriptions(self) -> Sequence[str]:
        return (
            "enter: step once",
            "a: toggle auto-run",
            "m: inject todo",
            "s: cycle generator",
            "g <key>: switch generator",
            "r: reset runtime",
            "h: show help",
            "q: quit",
        )

    def _command_prompt(self) -> str:
        auto_state = "ON" if self._auto else "OFF"
        return f"[auto={auto_state}] command> "


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
