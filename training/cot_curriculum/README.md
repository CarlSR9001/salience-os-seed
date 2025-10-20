# Chain-of-Thought Curriculum (Tier 1)

This package seeds structured chain-of-thought exemplars for the salience-native runtime. Each lesson is annotated with salience context, reasoning traces, and meta-lessons to accelerate reflection and self-supervised fine-tuning.

## Structure
```
training/cot_curriculum/
├── 00_basic_decomposition/
│   ├── math_word_problems.jsonl
│   ├── logic_puzzles.jsonl
│   └── metadata.yaml
├── 01_meta_reasoning/
│   ├── uncertainty_acknowledgment.jsonl
│   ├── self_correction.jsonl
│   └── assumption_questioning.jsonl
├── 02_verification_driven/
│   ├── predict_then_check.jsonl
│   ├── contradiction_detection.jsonl
│   └── consistency_checking.jsonl
├── 03_memory_integration/
│   ├── retrieve_before_answer.jsonl
│   ├── fact_synthesis.jsonl
│   └── hypothesis_formation.jsonl
└── 04_failure_recovery/
    ├── confusion_diagnosis.jsonl
    ├── strategy_switching.jsonl
    └── scope_reduction.jsonl
```

## JSONL Schema
Each entry is a JSON object with:
- `task`: short natural language prompt.
- `salience_context`: mapping with `uncertainty`, `novelty`, `aim`, `coherence`, `drag` estimates.
- `reasoning_trace`: ordered list of scratchpad steps.
- `answer`: concise final response.
- `meta_lesson`: takeaway for reflection/pattern extraction.

Example:
```
{"task": "Estimate population of Tokyo in 2024", "salience_context": {"uncertainty": 0.7, "novelty": 0.4, "aim": 0.8}, "reasoning_trace": ["I need to estimate Tokyo's population. Let me check my uncertainty: 0.7 - quite high.", "This makes sense because populations change and I don't have 2024 data (cutoff Jan 2025).", "I should acknowledge this uncertainty explicitly rather than state a precise number.", "My knowledge: Tokyo metropolitan area was ~37-38M in 2020-2023.", "Reasonable estimate: 38M ± 2M with moderate confidence.", "Better to give a range than false precision."], "answer": "Tokyo metropolitan area: approximately 38 million (±2M), moderate confidence due to data recency", "meta_lesson": "High uncertainty → acknowledge limits, give ranges, express confidence level"}
```

## Metadata
Each lesson folder includes `metadata.yaml` capturing:
- `lesson`: human-readable summary.
- `targets`: skill focus (e.g., decomposition, verification).
- `difficulty`: `beginner` | `intermediate` | `advanced`.
- `notes`: optional references.

## Usage
- During curriculum ingestion, feed JSONL streams into `SalienceRuntime` via a dedicated training loop that records scratchpad traces and registers reasoning patterns.
- `tests/test_rlm.py` exercises salience-ranked spawn behaviour; add complementary tests ensuring curriculum examples integrate without format drift.
- Extend `docs/runtime_walkthrough.md` or a dedicated `docs/cot_curriculum.md` with guidelines for authoring new lessons.
