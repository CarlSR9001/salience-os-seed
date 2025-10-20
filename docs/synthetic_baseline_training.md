# Synthetic Baseline Checkpoint

This repository now ships with a minimal, fully synthetic corpus that can be used to
bootstrap a conversational checkpoint capable of polite English dialog, elementary
math, and tool hand-offs. The dataset intentionally remains tiny so it can be
retrained quickly or regenerated without touching any copyrighted material.

## Corpus Layout

```
data/local_benchmarks/synthetic_baseline_corpus.txt
  ├─ 28 short conversation snippets with math, summaries, and tool call patterns
  ├─ uses `<|system|>`, `<|user|>`, and `<|assistant|>` delimiters for clarity
  └─ covers calculator and weather tool responses via synthetic logs

data/local_benchmarks/synthetic_baseline_validation.txt
  └─ lightweight validation prompts mirroring the training distribution

data/local_benchmarks/synthetic_baseline_test.txt
  └─ optional test prompts for quick sanity checks
```

Because the files are so small (well under 10 kB combined), they are easy to
inspect or regenerate. When generating replacements, keep the same formatting so
they remain compatible with the ingestion pipeline.

## Training

To produce a fresh checkpoint, run:

```bash
python start.standard.py
```

The launcher is preconfigured to stream the synthetic corpus and emit a checkpoint
at `storage/proto_lm/synthetic_baseline.pt`. You can override the corpus or
checkpoint path via `salience_os_seed/training/run_corpus.py` if needed.

## Sampling & Evaluation

* `python -m salience_os_seed.tools.sample_synthetic_baseline` loads the checkpoint
  and prints a quick sample.
* `python eval.synthetic_baseline.py --checkpoint storage/proto_lm/synthetic_baseline.pt`
  computes simple metrics on the validation/test snippets and prints qualitative
  completions.

Both scripts expect the checkpoint to exist; if it has not been trained yet they
will raise a helpful error instructing you to run the training step first.
