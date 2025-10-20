"""Conversational CLI for emergent Salience Runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..telemetry import BUS, ParameterEvent, TelemetryEvent, render_parameter_event, render_training_event
from .filters import IngestionThresholds
from .session import ConversationConfig, ConversationSession, IngestionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SalienceOS emergent conversation loop")
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Directory of text files to bootstrap from (can be repeated)",
    )
    parser.add_argument(
        "--load-dv1",
        action="store_true",
        help="Preload every .txt/.md/.jsonl file under C:/DV1",
    )
    parser.add_argument("--chunk-size", type=int, default=2048, help="Ingestion chunk size (bytes)")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Autosave after this many accepted chunks during ingest",
    )
    parser.add_argument("--chunk-overlap", type=int, default=0, help="Overlap between successive chunks")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of chunks to evaluate per batch")
    parser.add_argument(
        "--max-ingest-chunks",
        type=int,
        default=None,
        help="Optional limit on number of chunks to ingest per corpus directory",
    )
    parser.add_argument(
        "--salience-filter",
        action="store_true",
        help="Enable salience-based filtering during ingestion",
    )
    parser.add_argument("--min-uncertainty", type=float, default=0.0, help="Minimum uncertainty to accept chunk")
    parser.add_argument("--min-novelty", type=float, default=0.0, help="Minimum novelty to accept chunk")
    parser.add_argument("--max-drag", type=float, default=1.5, help="Maximum drag allowed for chunk")
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Display ingestion progress (default)",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable ingestion progress output",
    )
    parser.add_argument("--autosave", type=str, default=None, help="Path to persist conversation state")
    parser.add_argument("--response-tokens", type=int, default=48, help="Maximum tokens in generated responses")
    parser.add_argument("--steps", type=int, default=None, help="Optional max number of exchanges before exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress meta reports")
    parser.add_argument("--adaptive-filter", action="store_true", help="Enable adaptive salience filtering")
    parser.add_argument("--adaptive-warmup", type=int, default=32)
    parser.add_argument("--adaptive-window", type=int, default=32)
    parser.add_argument("--adaptive-enable-ratio", type=float, default=0.3)
    parser.add_argument("--adaptive-disable-ratio", type=float, default=0.6)
    parser.add_argument("--adaptive-cooldown", type=int, default=32)
    parser.add_argument("--behavior-scaffolds", type=str, default=None, help="Path to behavior scaffold JSONL")
    parser.add_argument("--no-learning", action="store_true", help="Disable online learning during chat")
    parser.add_argument("--telemetry", action="store_true", help="Stream telemetry events to stderr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = IngestionThresholds(
        enabled=args.salience_filter,
        min_uncertainty=args.min_uncertainty,
        min_novelty=args.min_novelty,
        max_drag=args.max_drag,
    )
    ingestion_cfg = IngestionConfig(
        thresholds=thresholds,
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        max_chunks=args.max_ingest_chunks,
        progress_enabled=args.progress,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    if args.adaptive_filter:
        ingestion_cfg.adaptive = ingestion_cfg.adaptive.__class__(
            enabled=True,
            warmup_chunks=args.adaptive_warmup,
            evaluation_window=args.adaptive_window,
            enable_ratio=args.adaptive_enable_ratio,
            disable_ratio=args.adaptive_disable_ratio,
            cooldown_chunks=args.adaptive_cooldown,
        )
    config = ConversationConfig(
        response_tokens=args.response_tokens,
        auto_save_path=args.autosave,
        ingestion=ingestion_cfg,
        learning_enabled=not args.no_learning,
        behavior_scaffolds_path=args.behavior_scaffolds,
    )
    session = ConversationSession(config=config)
    unsubscribe = None
    if args.telemetry:
        def _telemetry_sink(event: TelemetryEvent) -> None:
            if isinstance(event, ParameterEvent):
                print(render_parameter_event(event), file=sys.stderr)
            elif event.type == "training/step":
                for line in render_training_event(event):
                    print(line, file=sys.stderr)
            elif event.type == "ingestion/chunk":
                from ..telemetry import render_ingestion_event

                print(render_ingestion_event(event), file=sys.stderr)

        unsubscribe = BUS.subscribe(_telemetry_sink)

    corpus_paths = []
    if args.corpus:
        corpus_paths.extend(args.corpus)
    if args.load_dv1:
        corpus_paths.append(os.path.join("C:/DV1"))

    def progress_cb(payload: dict) -> None:
        if not args.progress:
            return
        meta = payload["metadata"]
        stats = payload["stats"]
        estimate = payload.get("estimate", {})
        eta = estimate.get("eta", "unknown")
        total_chunks = estimate.get("chunks", 0) or 1
        pct = min(100.0, 100.0 * stats["chunks_processed"] / total_chunks)
        line = (
            f"[ingest] {pct:5.1f}% | file {meta['file_index']}/{estimate.get('files', '?')} | "
            f"chunk {meta['chunk_index']}/{meta['chunk_total']} | accepted={stats['chunks_accepted']} "
            f"rejected={stats['chunks_rejected']} | eta={eta}"
        )
        print(line, end="\r", file=sys.stderr, flush=True)

    for path in corpus_paths:
        print(f"[bootstrapping] ingesting corpus at {path}")
        stats = session.ingest_directory(path, max_chunks=args.max_ingest_chunks, progress_cb=progress_cb)
        if args.progress:
            print("", file=sys.stderr)
        print(
            f"[ingest] completed {stats.chunks_processed} chunks "
            f"(accepted={stats.chunks_accepted}, rejected={stats.chunks_rejected})"
        )

    print("SalienceOS Emergent Chat — type 'exit' to quit")
    exchanges = 0
    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        metrics = session.process_user_input(user_input)
        snapshot = session.generate_response()
        if not args.quiet:
            print(f"[meta] {snapshot.meta_report}")
        print(f"agent> {snapshot.response}")
        exchanges += 1
        if args.steps is not None and exchanges >= args.steps:
            break
    if args.autosave:
        saved = session.save_state(args.autosave)
        print(f"State saved to {saved}")
    if unsubscribe:
        unsubscribe()


if __name__ == "__main__":
    main()
