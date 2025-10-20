"""Command helpers for running salience-filtered ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional

from ..conversation.filters import IngestionThresholds
from ..conversation.session import (
    AdaptiveFilteringConfig,
    ConversationConfig,
    ConversationSession,
    IngestionConfig,
)
from ..telemetry import BUS, ParameterEvent, TelemetryEvent, render_parameter_event, render_training_event


def ingest_directory(
    root: str | Path,
    *,
    autosave: str | None = None,
    ingestion: IngestionConfig | None = None,
    progress: bool = True,
    learning_enabled: bool = True,
    behavior_scaffolds: str | None = None,
) -> Mapping[str, int]:
    """Run ingestion for ``root`` and return stats mapping."""

    ingestion_cfg = ingestion or IngestionConfig(progress_enabled=progress)
    if not progress:
        ingestion_cfg = replace(ingestion_cfg, progress_enabled=False)
    config = ConversationConfig(
        auto_save_path=autosave,
        ingestion=ingestion_cfg,
        learning_enabled=learning_enabled,
        behavior_scaffolds_path=behavior_scaffolds,
    )
    session = ConversationSession(config=config)
    stats = session.ingest_directory(str(root), progress_cb=_make_progress_cb(progress))
    return stats.as_mapping()


def _make_progress_cb(enabled: bool):
    if not enabled:
        return None

    def _callback(payload: Mapping[str, object]) -> None:
        meta = payload.get("metadata", {})
        stats = payload.get("stats", {})
        estimate = payload.get("estimate", {})
        files = estimate.get("files", "?")
        total_chunks = estimate.get("chunks", 1) or 1
        processed = stats.get("chunks_processed", 0)
        pct = min(100.0, 100.0 * processed / total_chunks)
        eta = estimate.get("eta", "unknown")
        line = (
            f"[ingest] {pct:5.1f}% | file {meta.get('file_index', '?')}/{files} | "
            f"chunk {meta.get('chunk_index', '?')}/{meta.get('chunk_total', '?')} | "
            f"accepted={stats.get('chunks_accepted', 0)} rejected={stats.get('chunks_rejected', 0)} | eta={eta}"
        )
        print(line, end="\r", file=sys.stderr, flush=True)

    return _callback


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run salience-filtered ingestion on a directory")
    parser.add_argument("root", help="Directory containing corpus files")
    parser.add_argument("--autosave", type=str, default=None, help="Path to autosave conversation state")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size in bytes")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Autosave every N accepted chunks")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional limit on processed chunks")
    parser.add_argument("--chunk-overlap", type=int, default=0, help="Overlap between successive chunks")
    parser.add_argument("--batch-size", type=int, default=1, help="Process this many chunks per evaluation batch")
    parser.add_argument("--salience-filter", action="store_true", help="Enable salience thresholds")
    parser.add_argument("--min-uncertainty", type=float, default=0.0)
    parser.add_argument("--min-novelty", type=float, default=0.0)
    parser.add_argument("--max-drag", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    parser.add_argument("--output", type=str, default=None, help="Write stats JSON to path")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary print")
    parser.add_argument("--adaptive-filter", action="store_true", help="Enable adaptive filtering")
    parser.add_argument("--adaptive-warmup", type=int, default=32)
    parser.add_argument("--adaptive-window", type=int, default=32)
    parser.add_argument("--adaptive-enable-ratio", type=float, default=0.3)
    parser.add_argument("--adaptive-disable-ratio", type=float, default=0.6)
    parser.add_argument("--adaptive-cooldown", type=int, default=32)
    parser.add_argument("--behavior-scaffolds", type=str, default=None, help="Behavior scaffold JSONL path")
    parser.add_argument("--no-learning", action="store_true", help="Disable on-the-fly learning during ingest")
    parser.add_argument("--telemetry", action="store_true", help="Stream telemetry events during ingestion")
    return parser.parse_args(argv)


def _build_ingestion_config(args: argparse.Namespace) -> IngestionConfig:
    thresholds = IngestionThresholds(
        enabled=args.salience_filter,
        min_uncertainty=args.min_uncertainty,
        min_novelty=args.min_novelty,
        max_drag=args.max_drag,
    )
    adaptive = AdaptiveFilteringConfig()
    if args.adaptive_filter:
        adaptive = AdaptiveFilteringConfig(
            enabled=True,
            warmup_chunks=args.adaptive_warmup,
            evaluation_window=args.adaptive_window,
            enable_ratio=args.adaptive_enable_ratio,
            disable_ratio=args.adaptive_disable_ratio,
            cooldown_chunks=args.adaptive_cooldown,
        )
    return IngestionConfig(
        thresholds=thresholds,
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        max_chunks=args.max_chunks,
        progress_enabled=not args.no_progress,
        adaptive=adaptive,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    unsubscribe = None
    if args.telemetry:
        def _telemetry_sink(event: TelemetryEvent) -> None:
            if isinstance(event, ParameterEvent):
                print(render_parameter_event(event), file=sys.stderr)
            elif event.type == "training/step":
                for line in render_training_event(event):
                    print(line, file=sys.stderr)

        unsubscribe = BUS.subscribe(_telemetry_sink)
    stats = ingest_directory(
        args.root,
        autosave=args.autosave,
        ingestion=_build_ingestion_config(args),
        progress=not args.no_progress,
        learning_enabled=not args.no_learning,
        behavior_scaffolds=args.behavior_scaffolds,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(stats, indent=2))
    if unsubscribe:
        unsubscribe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
