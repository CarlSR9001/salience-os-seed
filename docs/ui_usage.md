# SalienceOS Seed UI Usage Guide

## Prerequisites
- Python 3.13+ (matching the environment used for development).
- Runtime root on `PYTHONPATH`, e.g. in PowerShell:
  ```powershell
  $env:PYTHONPATH = "C:\dv1\UAE-Model"
  ```
- Optional: `rich` for enhanced terminal rendering (install via `python -m pip install rich`).

## Launching the Dashboard
Run the CLI entry point from the repository root:
```powershell
python -m salience_os_seed.runtime.ui.cli --generator baseline
```

### Useful Flags
- `--generator {baseline,spiky,draggy}`: choose a synthetic state profile.
- `--auto`: start in auto-run mode (continuous stepping).
- `--interval 0.5`: adjust auto-run refresh interval (seconds).
- `--rich`: force rich rendering (requires `rich`).
- `--plain`: force plain text rendering even if `rich` is installed.

## Controls
- **Enter**: single step (when auto-run disabled).
- **a**: toggle auto-run.
- **m**: inject a sample todo into structured memory.
- **s**: cycle the active state generator.
- **g `<key>`**: switch directly to a generator by key (see the "Generators" panel for options).
- **r**: reset the runtime (clears history and restarts the orchestrator).
- **h** or **?**: show the command palette in the message feed.
- **q**: quit the dashboard.

## Observing Outputs
Each step surfaces:
- Meta-state self-report line with generator details and auto-run state.
- Controller decision tuple, score, hysteresis delta, cooldown.
- Scheduler event list, budget ratio, and sparklines for budget + verification trends.
- Salience channel magnitudes (top contributors).
- Recent step timeline with verification outcomes.
- Todos table showing structured memory edits, plus maintenance and experiment summaries.
- Command legend, generator roster, and rolling message feed so you always know what changed.

Rich mode renders multi-panel layouts; plain mode mirrors the same structure with wrapped tables and headings.
