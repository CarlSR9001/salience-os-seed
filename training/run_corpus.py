"""Utility to run multi-epoch corpus training with the ProtoLanguageModel.

This script streams a text corpus (TinyStories-scale) through the
``ProtoLanguageModel`` in multiple epochs, applying patience-based early
stopping when average epoch loss stops improving.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Iterable, Iterator, List, Optional

from ..conversation.filters import IngestionThresholds
from ..conversation.session import ConversationConfig, ConversationSession, IngestionConfig
from ..proto_lm.trainer import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-epoch training on a text corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to UTF-8 text corpus (e.g., standard/TinyStories-train.txt)")
    parser.add_argument("--epochs", type=int, default=0, help="Maximum epochs to run (0 = infinite until early-stop)")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Approximate character budget per training segment")
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="Number of stories to buffer and shuffle before packing into segments (0 = no shuffling)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for shuffling")
    parser.add_argument("--patience", type=int, default=3, help="Epochs without improvement before stopping (0 disables early stop)")
    parser.add_argument("--min-delta", type=float, default=0.1, help="Minimum improvement in mean loss to reset patience")
    parser.add_argument("--log-every", type=int, default=1000, help="Print progress every N training updates (0 disables)")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("storage/proto_lm/tinystories.pt"), help="File path for checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Steps between checkpoint saves (0 disables periodic saves)")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint-path if it exists without archiving")
    parser.add_argument("--salience-filter", action="store_true", help="Enable the salience ingestion filter before training")
    parser.add_argument("--min-uncertainty", type=float, default=0.0, help="Minimum uncertainty required to accept a segment when the filter is enabled")
    parser.add_argument("--min-novelty", type=float, default=0.0, help="Minimum novelty required to accept a segment when the filter is enabled")
    parser.add_argument("--max-drag", type=float, default=1.0, help="Maximum drag allowed to accept a segment when the filter is enabled")
    return parser.parse_args()


def pack_segments(stories: Iterable[str], chunk_size: int) -> Iterator[str]:
    buffer: List[str] = []
    total = 0
    for story in stories:
        if not story:
            continue
        buffer.append(story)
        total += len(story) + 1
        if total >= chunk_size:
            yield "\n".join(buffer)
            buffer.clear()
            total = 0
    if buffer:
        yield "\n".join(buffer)


def iter_segments(path: Path, *, chunk_size: int, shuffle_buffer: int, rng: random.Random) -> Iterator[str]:
    if shuffle_buffer <= 0:
        with path.open("r", encoding="utf-8") as handle:
            yield from pack_segments((line.strip() for line in handle), chunk_size)
        return

    buffer: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            buffer.append(stripped)
            if len(buffer) >= shuffle_buffer:
                rng.shuffle(buffer)
                yield from pack_segments(buffer, chunk_size)
                buffer.clear()
    if buffer:
        rng.shuffle(buffer)
        yield from pack_segments(buffer, chunk_size)


def build_session(args: argparse.Namespace) -> ConversationSession:
    ingestion = IngestionConfig(
        chunk_size=args.chunk_size,
        dedupe_enabled=False,
        allow_reingest_duplicates=True,
        progress_enabled=False,
        thresholds=IngestionThresholds(
            enabled=args.salience_filter,
            min_uncertainty=args.min_uncertainty,
            min_novelty=args.min_novelty,
            max_drag=args.max_drag,
        ),
    )
    training = TrainingConfig()
    training.seed = args.seed
    training.checkpoint_path = str(args.checkpoint_path)
    training.device = "auto"

    config = ConversationConfig(lm=training, ingestion=ingestion)
    session = ConversationSession(config=config)
    if args.resume and args.checkpoint_path.exists():
        session.proto_lm.load_checkpoint(str(args.checkpoint_path))
    return session


def should_stop(epoch_losses: List[float], best_loss: float, min_delta: float, patience: int, stale_epochs: int) -> tuple[bool, float, int]:
    epoch_mean = fmean(epoch_losses) if epoch_losses else math.nan
    if math.isnan(epoch_mean):
        return False, best_loss, stale_epochs
    if epoch_mean + min_delta < best_loss:
        return False, epoch_mean, 0
    if patience <= 0:
        return False, best_loss, stale_epochs
    stale_epochs += 1
    return stale_epochs >= patience, best_loss, stale_epochs


def train_corpus(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    session = build_session(args)
    model = session.proto_lm
    print(f"Using device: {model.device}")

    corpus_bytes = args.corpus.stat().st_size
    print(
        f"Starting training on {args.corpus} (~{corpus_bytes / 1_000_000:.2f} MB)"
        f" with checkpoint {args.checkpoint_path}"
    )

    best_loss = float("inf")
    stale_epochs = 0
    target_epochs = args.epochs if args.epochs > 0 else math.inf
    epoch_index = 0

    filter_thresholds = session.config.ingestion.thresholds
    if filter_thresholds.enabled:
        print(
            "Salience filter enabled: min_uncertainty={:.3f} min_novelty={:.3f} max_drag={:.3f}".format(
                filter_thresholds.min_uncertainty,
                filter_thresholds.min_novelty,
                filter_thresholds.max_drag,
            )
        )

    while epoch_index < target_epochs:
        epoch_index += 1
        print(f"\n=== Epoch {epoch_index} (global step {model.step}) ===")
        epoch_start = perf_counter()
        losses: List[float] = []
        accepted_chunks = 0
        rejected_chunks = 0
        meta_snapshot = session.runtime.meta_state.snapshot()
        memory_snapshot = session.runtime.memory.as_runtime_mapping()
        for step, segment in enumerate(iter_segments(args.corpus, chunk_size=args.chunk_size, shuffle_buffer=args.shuffle_buffer, rng=rng), start=1):
            digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:12]
            state = session._build_state(segment, speaker="corpus")
            accepts, readings = session._filter.evaluate(state, memory_snapshot, meta_snapshot)
            if not accepts:
                rejected_chunks += 1
                if args.log_every and (step == 1 or step % args.log_every == 0):
                    print(
                        "  step {local:>7} | global {global_step:>7} | status REJECT"
                        " | seg_len {seg_len:>5} | digest {digest} | novelty {novelty:.2f} | uncertainty {uncertainty:.2f} | drag {drag:.2f}".format(
                            local=step,
                            global_step=model.step,
                            seg_len=len(segment),
                            digest=digest,
                            novelty=float(readings.get("novelty", 0.0)),
                            uncertainty=float(readings.get("uncertainty", 0.0)),
                            drag=float(readings.get("drag", 0.0)),
                        )
                    )
                continue

            loss = model.training_step(segment)
            losses.append(loss)
            accepted_chunks += 1

            runtime_state = dict(state)
            runtime_state.setdefault("source", "corpus")
            runtime_state.setdefault("speaker", "corpus")
            runtime_state.setdefault("context_snippet", segment[:160])
            runtime_state.setdefault("segment_digest", digest)
            metrics = session.runtime.run_step(runtime_state)
            session._adaptive.track_runtime(metrics)
            gating = session._adaptive.last_gating_summary
            gate_decision = gating.truth_decision.decision if gating else "NONE"
            if args.log_every:
                if step == 1 or step % args.log_every == 0:
                    print(
                        "  step {local:>7} | global {global_step:>7} | loss {loss:8.4f}"
                        " | avg {avg:8.4f} | seg_len {seg_len:>5} | digest {digest}"
                        " | gate {gate} | novelty {novelty:.2f} | uncertainty {uncertainty:.2f} | drag {drag:.2f}".format(
                            local=step,
                            global_step=model.step,
                            loss=loss,
                            avg=fmean(losses),
                            seg_len=len(segment),
                            digest=digest,
                            gate=gate_decision,
                            novelty=float(readings.get("novelty", 0.0)),
                            uncertainty=float(readings.get("uncertainty", 0.0)),
                            drag=float(readings.get("drag", 0.0)),
                        )
                    )
            if args.checkpoint_interval and model.step % args.checkpoint_interval == 0:
                path = model.save_checkpoint(str(args.checkpoint_path))
                print(f"  ✔ checkpoint saved to {path}")

            meta_snapshot = session.runtime.meta_state.snapshot()
            memory_snapshot = session.runtime.memory.as_runtime_mapping()

        if not losses:
            print("No segments processed; stopping early.")
            break

        epoch_loss = fmean(losses)
        elapsed = perf_counter() - epoch_start
        print(
            "Epoch {epoch} stats: mean {mean:.4f} | median {med:.4f} | min {minv:.4f} | max {maxv:.4f}"
            " | updates {count} | accepted {accepted} | rejected {rejected} | elapsed {elapsed:.1f}s".format(
                epoch=epoch_index,
                mean=epoch_loss,
                med=median(losses),
                minv=min(losses),
                maxv=max(losses),
                count=len(losses),
                accepted=accepted_chunks,
                rejected=rejected_chunks,
                elapsed=elapsed,
            )
        )
        path = model.save_checkpoint(str(args.checkpoint_path))
        print(f"  ✔ checkpoint saved to {path}")

        stop, best_loss, stale_epochs = should_stop(losses, best_loss, args.min_delta, args.patience, stale_epochs)
        if stop:
            print(f"Early stopping triggered after {epoch_index} epochs (best loss {best_loss:.4f}).")
            break

    final_path = model.save_checkpoint(str(args.checkpoint_path))
    total_steps = model.step
    print(f"Training complete after {total_steps} updates. Final checkpoint: {final_path}")


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_corpus(args)


if __name__ == "__main__":
    main()
