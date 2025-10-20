"""Diagnostics helpers to inspect ingestion results and conversation health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional

from ..conversation.filters import IngestionThresholds
from ..conversation.session import ConversationConfig, ConversationSession, IngestionConfig


def run_diagnostics(
    corpus: str | Path,
    *,
    sample_prompt: str = "hello",
    max_response_tokens: int = 48,
    output: Optional[str] = None,
    ingestion: Optional[IngestionConfig] = None,
) -> Mapping[str, object]:
    cfg = ConversationConfig(
        response_tokens=max_response_tokens,
        ingestion=ingestion or IngestionConfig(progress_enabled=False),
    )
    session = ConversationSession(config=cfg)
    stats = session.ingest_directory(str(corpus), progress_cb=None)
    step = session.proto_lm.step
    vocab_size = session.proto_lm.vocab.size()
    sample = session.proto_lm.sample(sample_prompt, max_tokens=8)
    metrics = session.process_user_input(sample_prompt)
    snapshot = session.generate_response()
    result = {
        "ingestion": stats.as_mapping(),
        "proto_lm": {
            "step": step,
            "vocab_size": vocab_size,
            "sample": sample,
        },
        "conversation": {
            "last_meta": metrics.meta_report,
            "response": snapshot.response,
            "todos": list(snapshot.todos),
        },
    }
    if output:
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ingestion and conversation diagnostics")
    parser.add_argument("corpus", help="Directory to ingest")
    parser.add_argument("--sample-prompt", default="hello", help="Prompt to probe conversation")
    parser.add_argument("--response-tokens", type=int, default=48, help="Max tokens for generated response")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--no-progress", action="store_true", help="Disable ingestion progress")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--salience-filter", action="store_true")
    parser.add_argument("--min-uncertainty", type=float, default=0.0)
    parser.add_argument("--min-novelty", type=float, default=0.0)
    parser.add_argument("--max-drag", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    ingestion_cfg = IngestionConfig(
        thresholds=IngestionThresholds(
            enabled=args.salience_filter,
            min_uncertainty=args.min_uncertainty,
            min_novelty=args.min_novelty,
            max_drag=args.max_drag,
        ),
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        max_chunks=args.max_chunks,
        progress_enabled=not args.no_progress,
    )
    result = run_diagnostics(
        args.corpus,
        sample_prompt=args.sample_prompt,
        max_response_tokens=args.response_tokens,
        output=args.output,
        ingestion=ingestion_cfg,
    )
    if not args.output:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
