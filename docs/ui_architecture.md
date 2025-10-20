# SalienceOS Seed Interaction Layer – Architecture Notes

## Goals
- Provide an interactive terminal UI to observe salience dynamics in real time.
- Keep dependencies lean (standard library + optional `rich` fallback) so the seed remains easy to run.
- Support command-driven interaction for injecting synthetic events, toggling stochastic knobs, and stepping the runtime loop.

## Key Components
- **`runtime/driver.py`** – Cohesive interface atop `SalienceRuntime`. Responsibilities:
  - Owns a `SalienceRuntime` instance and handles lifecycle (initialisation, stepping, reset).
  - Manages synthetic state generation via pluggable providers.
  - Exposes high-level methods (`step()`, `apply_memory_verb()`, `toggle_tool()`) used by the UI.
  - Stores recent `RuntimeMetrics` in a ring buffer for inspection.

- **`runtime/state_gen.py`** – Synthetic state generators generating `state` mappings:
  - Deterministic baseline generator (static logits/tokens).
  - Stochastic generator introducing salience spikes (uncertainty, novelty, drag toggles) for demos.
  - Exposes protocol `StateGenerator` with `next_state()` and `describe()`.

- **`runtime/ui/dashboard.py`** – UI rendering layer using a simple reactive loop:
  - Renders sections: headline (meta report + generator details), controller decision, runtime status with trend sparklines, scratchpad, salience channels, history timeline, todos, maintenance/experiments, command legend, generator roster, and rolling status messages.
  - Accepts keyboard commands (`[enter]` = single step, `a` = auto-run toggle, `m` = mutate memory, `s` = cycle generator, `g <key>` = jump to generator, `r` = reset runtime, `h` = help, `q` = quit).
  - Uses `rich` if installed; falls back to structured plain text formatting otherwise (shared command surface + summaries).

- **`runtime/ui/cli.py`** – Entry point executed via `python -m salience_os_seed.runtime.ui.cli`.
  - Parses CLI args (`--generator`, `--auto`, `--steps`, `--rich` override).
  - Bootstraps driver + dashboard and runs the control loop.

## Data Flow
```
StateGenerator.next_state() -> RuntimeDriver.step(state)
  -> SalienceRuntime.run_step()
     -> RuntimeMetrics + StructuredMemory updates
  <- RuntimeDriver stores metrics/ring buffer
UI refresh pulls from driver.snapshot()
Commands mutate driver/generator/runtime state
```

## Command Surface
- `enter`: single step.
- `a`: toggle auto-run (stream updates until paused).
- `m`: inject a todo into structured memory.
- `s`: cycle state generator profile.
- `g <key>`: jump directly to a specific generator.
- `r`: reset the runtime loop and history.
- `h` / `?`: surface the command palette in the message feed.
- `q`: exit.

## Testing Plan
- Unit tests for `RuntimeDriver` (buffering, generator integration).
- UI smoke test ensuring CLI boots in headless mode and renders at least one frame (captured output).
- Deterministic generator tests verifying emitted salience cues.
