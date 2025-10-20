"""Lightweight telemetry event bus shared across subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Iterator, List, Mapping


@dataclass(frozen=True)
class TelemetryEvent:
    """Generic event payload published to subscribers."""

    type: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterEvent(TelemetryEvent):
    """Telemetry event describing parameter updates."""

    type: str = "parameters/update"


class TelemetryBus:
    """Thread-safe pub/sub bus for telemetry events."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[TelemetryEvent], None]] = []
        self._lock = Lock()

    def subscribe(self, callback: Callable[[TelemetryEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return _unsubscribe

    def publish(self, event: TelemetryEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # Telemetry should never crash the main loop; swallow subscriber errors.
                continue


BUS = TelemetryBus()


ANSI_RESET = "\x1b[0m"
ANSI_GREEN = "\x1b[32m"
ANSI_RED = "\x1b[31m"
ANSI_BLUE = "\x1b[34m"
ANSI_CYAN = "\x1b[36m"
ANSI_YELLOW = "\x1b[33m"


def colour_text(text: str, colour: str) -> str:
    return f"{colour}{text}{ANSI_RESET}"


def render_parameter_event(event: ParameterEvent) -> str:
    payload = event.payload
    delta = int(payload.get("delta", 0))
    colour = ANSI_GREEN if delta > 0 else ANSI_RED if delta < 0 else ANSI_BLUE
    delta_str = f"{delta:+d}" if isinstance(delta, int) else str(delta)
    coloured_delta = colour_text(delta_str, colour)
    total = payload.get("total", "?")
    step = payload.get("step", "?")
    return f"[telemetry] step={step} params={total} Δ={coloured_delta}"


def render_training_event(event: TelemetryEvent) -> Iterator[str]:
    payload = event.payload
    step = payload.get("step", "?")
    loss = payload.get("loss", "?")
    total = payload.get("parameter_total", "?")
    yield f"[telemetry] step={step} loss={loss:.4f} params={total}" if isinstance(loss, float) else f"[telemetry] step={step} loss={loss} params={total}"
    grads = payload.get("grads", [])
    if isinstance(grads, list) and grads:
        for name, norm in grads[:5]:
            colour = ANSI_GREEN if norm and norm > 0 else ANSI_BLUE
            if isinstance(name, str):
                yield f"    grad {name}: {colour_text(f'{norm:.4f}' if isinstance(norm, float) else str(norm), colour)}"


def render_ingestion_event(event: TelemetryEvent) -> str:
    payload = event.payload
    accepted = bool(payload.get("accepted", False))
    path = payload.get("path", "?")
    chunk_index = payload.get("chunk_index", "?")
    chunk_total = payload.get("chunk_total", "?")
    stats = payload.get("stats", {})
    accepted_count = stats.get("chunks_accepted", "?")
    rejected_count = stats.get("chunks_rejected", "?")
    colour = ANSI_GREEN if accepted else ANSI_RED
    status = "accepted" if accepted else "rejected"
    status_text = colour_text(status, colour)
    filter_state = payload.get("filter_enabled", False)
    filter_text = colour_text("filter:on" if filter_state else "filter:off", ANSI_CYAN)
    return (
        f"[telemetry] chunk {chunk_index}/{chunk_total} {status_text} "
        f"{filter_text} accepted={accepted_count} rejected={rejected_count} path={path}"
    )
