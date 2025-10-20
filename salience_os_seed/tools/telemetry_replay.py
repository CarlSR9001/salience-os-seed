"""Replay telemetry logs to visualise salience evolution and controller choices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPARK_BARS = "▁▂▃▄▅▆▇█"


class SessionReplay:
    """Aggregate telemetry events and render a textual replay."""

    def __init__(self) -> None:
        self._histories: List[Tuple[int, Mapping[str, Mapping[str, object]]]] = []
        self._decisions: List[Tuple[int, Mapping[str, object]]] = []

    def ingest(self, event: Mapping[str, object]) -> None:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            return
        if event_type == "runtime/salience_histogram":
            step = int(payload.get("step", -1))
            histograms = payload.get("histograms", {})
            if isinstance(histograms, Mapping):
                self._histories.append((step, histograms))
        elif event_type == "runtime/controller_trace":
            step = int(payload.get("step", -1))
            self._decisions.append((step, payload))

    def finalise(self) -> None:
        self._histories.sort(key=lambda item: item[0])
        self._decisions.sort(key=lambda item: item[0])

    def render(self, *, max_steps: Optional[int] = None, sensors: Optional[Sequence[str]] = None) -> None:
        self.finalise()
        hist_cursor = 0
        last_hist: Mapping[str, Mapping[str, object]] = {}
        last_salience: Dict[str, float] = {}
        rendered = 0
        for step, payload in self._decisions:
            while hist_cursor < len(self._histories) and self._histories[hist_cursor][0] <= step:
                last_hist = self._histories[hist_cursor][1]
                hist_cursor += 1
            decision = payload.get("decision", {})
            action = decision.get("action", {}) if isinstance(decision, Mapping) else {}
            score = decision.get("score") if isinstance(decision, Mapping) else None
            hysteresis = decision.get("hysteresis_delta") if isinstance(decision, Mapping) else None
            cooldown = decision.get("cooldown", {}) if isinstance(decision, Mapping) else {}
            executed = payload.get("executed")
            verification = payload.get("verification_passed")
            print(_format_header(step, action, score, hysteresis, cooldown, executed, verification))
            salience = payload.get("salience", {})
            if not isinstance(salience, Mapping):
                salience = {}
            filtered_sensors = set(sensor.lower() for sensor in sensors) if sensors else None
            for sensor_name in sorted(salience):
                if filtered_sensors and sensor_name.lower() not in filtered_sensors:
                    continue
                value = _safe_float(salience.get(sensor_name))
                hist = last_hist.get(sensor_name, {}) if isinstance(last_hist, Mapping) else {}
                line = _format_sensor_line(sensor_name, value, hist, last_salience.get(sensor_name))
                print(line)
                last_salience[sensor_name] = value if isinstance(value, float) else last_salience.get(sensor_name, 0.0)
            bids = payload.get("auction", [])
            if isinstance(bids, Iterable) and not isinstance(bids, (str, bytes)):
                bid_line = _format_bids(bids)
                if bid_line:
                    print(bid_line)
            print()
            rendered += 1
            if max_steps is not None and rendered >= max_steps:
                break


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_header(
    step: int,
    action: Mapping[str, object],
    score: Optional[object],
    hysteresis: Optional[object],
    cooldown: Mapping[str, object],
    executed: Optional[object],
    verification: Optional[object],
) -> str:
    operator = action.get("operator", "?") if isinstance(action, Mapping) else "?"
    depth = action.get("cot_depth", "?") if isinstance(action, Mapping) else "?"
    patch = action.get("patch", "?") if isinstance(action, Mapping) else "?"
    score_str = f"score={float(score):+.3f}" if isinstance(score, (int, float)) else "score=?"
    hysteresis_str = (
        f"Δ={float(hysteresis):+.3f}" if isinstance(hysteresis, (int, float)) else "Δ=?"
    )
    cooldown_prev = cooldown.get("previous") if isinstance(cooldown, Mapping) else None
    cooldown_curr = cooldown.get("current") if isinstance(cooldown, Mapping) else None
    cooldown_str = (
        f"cooldown {int(cooldown_prev)}→{int(cooldown_curr)}"
        if isinstance(cooldown_prev, (int, float)) and isinstance(cooldown_curr, (int, float))
        else "cooldown ?"
    )
    executed_str = "executed" if executed else "skipped"
    if verification is True:
        verification_str = "verification=pass"
    elif verification is False:
        verification_str = "verification=fail"
    else:
        verification_str = "verification=?"
    return (
        f"Step {int(step):>4} | op={operator} depth={depth} patch={patch} "
        f"{score_str} {hysteresis_str} {cooldown_str} {executed_str} {verification_str}"
    )


def _format_sensor_line(
    name: str,
    value: Optional[float],
    hist: Mapping[str, object],
    previous: Optional[float],
) -> str:
    value_str = f"{value:+.3f}" if isinstance(value, float) else "   ?"
    delta = None
    if isinstance(value, float) and isinstance(previous, float):
        delta = value - previous
    delta_str = f"Δ={delta:+.3f}" if isinstance(delta, float) else "Δ=?"
    bins = hist.get("bins") if isinstance(hist, Mapping) else None
    hist_str = _render_histogram(bins) if isinstance(bins, Sequence) else ""
    pointer = _render_pointer(value, hist)
    return f"  {name:<12} {value_str} {delta_str} {pointer} {hist_str}".rstrip()


def _render_histogram(bins: Sequence[Mapping[str, object]]) -> str:
    counts: List[int] = []
    for bucket in bins:
        count = bucket.get("count") if isinstance(bucket, Mapping) else None
        if isinstance(count, (int, float)):
            counts.append(int(count))
    if not counts:
        return ""
    max_count = max(counts) or 1
    spark = "".join(_SPARK_BARS[min(len(_SPARK_BARS) - 1, int(round(c / max_count * (len(_SPARK_BARS) - 1))))] for c in counts)
    return f"hist:{spark}"


def _render_pointer(value: Optional[float], hist: Mapping[str, object], width: int = 24) -> str:
    if not isinstance(value, float):
        return "".ljust(width)
    min_val = hist.get("min") if isinstance(hist, Mapping) else None
    max_val = hist.get("max") if isinstance(hist, Mapping) else None
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        return "".ljust(width)
    if max_val <= min_val:
        pos = width // 2
    else:
        ratio = (value - float(min_val)) / (float(max_val) - float(min_val))
        ratio = max(0.0, min(1.0, ratio))
        pos = int(round(ratio * (width - 1)))
    chars = ["─" for _ in range(width)]
    if 0 <= pos < width:
        chars[pos] = "▲"
    return "".join(chars)


def _format_bids(bids: Iterable[object]) -> str:
    formatted: List[str] = []
    for entry in bids:
        if not isinstance(entry, Mapping):
            continue
        action = entry.get("action")
        score = entry.get("score")
        if isinstance(score, (int, float)):
            score_str = f"{float(score):+.3f}"
        else:
            score_str = "?"
        formatted.append(f"{action}:{score_str}")
    if not formatted:
        return ""
    return "  bids: " + ", ".join(formatted)


def load_events(path: Optional[Path]) -> Iterable[Mapping[str, object]]:
    handle = sys.stdin if path is None else path.open("r", encoding="utf-8")
    try:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                yield event
    finally:
        if handle is not sys.stdin:
            handle.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", nargs="?", type=Path, help="Telemetry JSONL file. Defaults to stdin when omitted.")
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="Limit the number of decision steps rendered.")
    parser.add_argument(
        "--sensor",
        action="append",
        dest="sensors",
        help="Filter output to the provided sensor name (can be supplied multiple times).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    replay = SessionReplay()
    for event in load_events(args.logfile):
        replay.ingest(event)
    replay.render(max_steps=args.max_steps, sensors=args.sensors)


if __name__ == "__main__":
    main()
