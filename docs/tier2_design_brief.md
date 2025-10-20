# Tier 2 Design Brief – Memory Maintenance, Self-Experiments, Episodic Memory

## Memory Maintenance (core/memory/maintenance.py)
- **Purpose**: Prevent salience drag from unbounded structured memory growth.
- **Responsibilities**:
  - `should_cleanup(salience, memory)` – trigger when `drag > 0.5`, memory size exceeds thresholds, or repeated verification failures align with high drag.
  - `archive_low_roi_facts(memory, threshold=0.1)` – move stale facts to cold storage (e.g., JSONL under `storage/memory_archive/`).
  - `merge_redundant_entries(memory)` – detect similar facts via cosine similarity / Jaccard and merge into summaries.
  - `prune_failed_hypotheses(memory)` – track verification outcomes; remove hypotheses failing > N times.
  - `summarize_old_context(memory, age_threshold=1000)` – compress older transcripts into summary entries.
- **Integration**:
  - Invoke post-VERIFY or via scheduler event when drag spike detected.
  - Provide hooks for logging (telemetry + optional callbacks).
- **Testing**:
  - Unit tests with synthetic memory snapshots verifying archive/prune operations.
  - Integration test ensuring drag reduction after cleanup.

## Self-Directed Experiments (core/ideas/experiments.py)
- **Purpose**: Allow runtime to explore parameter tweaks with measurable outcomes.
- **Data Structures**:
  - `SelfExperiment` dataclass (hypothesis, parameter overrides, metrics, duration, results, conclusion).
  - `ExperimentDispatcher` orchestrates proposal → execution → analysis.
- **API**:
  - `propose_experiment(salience, meta)` – look for performance gaps (e.g., low verification success) and suggest parameter variations.
  - `run_experiment(experiment)` – apply temporary overrides (e.g., S′ weights, scheduler thresholds) for N steps; collect metrics.
  - `analyze_results(experiment)` – produce statistical summary + natural language conclusion.
- **Safety**:
  - Parameter allowlist, revert overrides on completion/failure.
  - Budget guardrails (max concurrent experiments, total steps).
- **Testing**:
  - Simulated experiments verifying metrics collected and overrides revert.
  - Ensure analysis returns structured report.

## Episodic Memory (core/meta/episodic.py)
- **Purpose**: Store summaries of past episodes to support reflection and failure mode recognition.
- **Structures**:
  - `Episode` dataclass (id, task_type, salience_profile, actions_taken, outcome, scratchpad_summary, lessons_learned).
  - `EpisodicStore` managing append, retrieval by salience similarity, and serialization to disk.
- **Features**:
  - `record_episode(metrics, scratchpad, verifier_outcome)` after each high-level interaction.
  - `retrieve_similar(salience_profile, top_k)` using cosine similarity over salience vectors.
  - `summarize_lessons()` for reflective reporting.
- **Integration**:
  - Hook into `SalienceRuntime.run_step()` after decisions; optionally triggered by scheduler.
  - Provide interface for RLM to fetch past episodes during spawn planning.
- **Testing**:
  - Unit tests for serialization/deserialization, similarity retrieval, and summary generation.

## Roadmap Links
- Implement memory maintenance first (prerequisite for stable drag metrics).
- Integrate self-experiments leveraging RLM tool `spawn` pathways for experimentation tasks.
- Add episodic store to support Tier 3 telemetry and visualization (future `SalienceLogger`).
