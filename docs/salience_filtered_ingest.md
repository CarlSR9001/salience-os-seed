# Salience-Filtered Corpus Ingestion – Design Notes

## Objectives
- Filter incoming training chunks using the existing salience sensors so the proto language model receives higher-quality signals.
- Provide transparent progress reporting (files scanned, chunks accepted/rejected, ETA) during large ingests.
- Enable periodic autosave checkpoints so long-running ingests can resume after interruption.
- Keep the implementation modular and easily adjustable (thresholds, sensor selection, batching).

## Architecture Overview
1. **Corpus Scanning** (`ingestion/reader.py`)
   - Extend `CorpusReader` to emit metadata: file index, total files, chunk index, chunk count.
   - Offer `estimate_work()` to pre-compute totals for progress reporting.

2. **Salience Evaluation** (`conversation/filters.py` new module)
   - Instantiate `SensorBank.default_bank()`.
   - Build a lightweight state stub from each chunk and existing meta snapshot.
   - Compute salience vector and decide acceptance using configurable thresholds (e.g., `uncertainty >= u_min`, `drag <= drag_max`, `novelty >= novelty_min`).
   - Emit both decision and raw readings for logging.

3. **Ingestion Pipeline** (`conversation/session.py`)
   - Replace direct call to `training_step()` with a loop that:
     1. Requests chunk metadata from `CorpusReader`.
     2. Evaluates salience; if rejected, skip training but still update progress counters.
     3. On acceptance, call `training_step()` and track statistics.
   - Add hooks for `on_progress` callbacks to enable CLI progress updates.
   - Trigger `save_state()` every N accepted chunks (configurable) and on completion.

4. **CLI Enhancements** (`conversation/cli.py`)
   - CLI flags:
     - `--salience-filter` to enable filtering.
     - Threshold overrides (`--min-uncertainty`, `--min-novelty`, `--max-drag`).
     - `--progress` to display a textual progress bar with ETA (default on when filtering).
     - `--checkpoint-interval` for autosave frequency during ingest.
   - Render progress via simple textual bar (no external deps) showing files processed, accepted ratio, estimated remaining time.

5. **Configuration Objects**
   - Introduce `IngestionConfig` dataclass storing thresholds, chunk limits, autosave interval.
   - Plumb `ConversationConfig` to own an `ingestion` field so adjustments remain central.

## Data Flow
```
CorpusReader.stream() -> chunk, metadata
    -> SalienceFilter.evaluate(chunk) -> decision + readings
        -> if accepted: ProtoLanguageModel.training_step(chunk)
    -> ProgressTracker.update(metadata, decision)
    -> maybe autosave
```

## Extensibility
- Thresholds are stored in `IngestionConfig` and can be widened/narrowed without touching core logic.
- Sensor registry is swappable (pass a custom bank) so additional sensors can be included later.
- Progress tracker exposes raw counters for UI or logging backends.

## Testing Strategy
- Unit test filter decisions with synthetic chunks and mocked sensor readings.
- Integration test that ingest with filtering accepts fewer chunks than raw, updates progress counters, and autosaves at the configured interval (using temp directories).
- CLI smoke test verifying progress output occurs and thresholds flow through to the session.
