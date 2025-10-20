"""Unified runner combining corpus training, dashboard, and chat interface."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import threading
import time
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional

from ..conversation.session import ConversationConfig, ConversationSession, IngestionConfig
from ..proto_lm.trainer import TrainingConfig
from ..runtime.ui.web_dashboard import run_dashboard_from_session
from ..telemetry import BUS, TelemetryEvent, render_ingestion_event, render_training_event
from ..training.run_corpus import iter_segments


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _render_gate(summary: Optional[object]) -> str:
    if summary is None:
        return "NONE"
    try:
        decision = getattr(summary, "truth_decision", None)
        if decision is None:
            return "UNKNOWN"
        return str(getattr(decision, "decision", "UNKNOWN"))
    except Exception:
        return "UNKNOWN"


def _default_ingestion_config(chunk_size: int) -> IngestionConfig:
    ingestion = IngestionConfig()
    ingestion.chunk_size = chunk_size
    ingestion.dedupe_enabled = False
    ingestion.allow_reingest_duplicates = True
    ingestion.progress_enabled = False
    return ingestion


def create_session(args: argparse.Namespace) -> ConversationSession:
    training_cfg = TrainingConfig()
    training_cfg.seed = args.seed
    training_cfg.device = args.device
    training_cfg.checkpoint_path = str(args.checkpoint)
    training_cfg.learning_rate = args.learning_rate
    config = ConversationConfig(
        lm=training_cfg,
        learning_enabled=not args.freeze_learning,
        ingestion=_default_ingestion_config(args.chunk_size),
        auto_save_path=args.auto_save_path,
    )
    session = ConversationSession(config=config)
    if args.resume:
        session.proto_lm.load_checkpoint(str(args.checkpoint))
    return session


def training_worker(
    *,
    session: ConversationSession,
    args: argparse.Namespace,
    lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    corpus_path = args.corpus
    if corpus_path is None:
        return
    rng = random.Random(args.seed)
    epochs_target = args.epochs if args.epochs > 0 else math.inf
    epoch_index = 0
    while epoch_index < epochs_target and not stop_event.is_set():
        epoch_index += 1
        _print_header(f"=== Epoch {epoch_index} (global step {session.proto_lm.step}) ===")
        losses: list[float] = []
        accepted = 0
        rejected = 0
        meta_snapshot: Mapping[str, float]
        memory_snapshot: Mapping[str, object]
        with lock:
            meta_snapshot = session.runtime.meta_state.snapshot()
            memory_snapshot = session.runtime.memory.as_runtime_mapping()
        for step, segment in enumerate(
            iter_segments(
                Path(corpus_path),
                chunk_size=args.chunk_size,
                shuffle_buffer=args.shuffle_buffer,
                rng=rng,
            ),
            start=1,
        ):
            if stop_event.is_set():
                break
            digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:12]
            with lock:
                state = session._build_state(segment, speaker="corpus")
                accepts, readings = session._filter.evaluate(state, memory_snapshot, meta_snapshot)
                if not accepts:
                    rejected += 1
                    if args.log_every and (step == 1 or step % args.log_every == 0):
                        print(
                            "  step {local:>7} | global {global_step:>7} | status REJECT"
                            " | seg_len {seg_len:>5} | digest {digest}"
                            " | novelty {novelty:.2f} | uncertainty {uncertainty:.2f} | drag {drag:.2f}".format(
                                local=step,
                                global_step=session.proto_lm.step,
                                seg_len=len(segment),
                                digest=digest,
                                novelty=float(readings.get("novelty", 0.0)),
                                uncertainty=float(readings.get("uncertainty", 0.0)),
                                drag=float(readings.get("drag", 0.0)),
                            )
                        )
                    continue
                loss = session.proto_lm.training_step(segment)
                losses.append(loss)
                accepted += 1
                runtime_state = dict(state)
                runtime_state.setdefault("source", "corpus")
                runtime_state.setdefault("speaker", "corpus")
                runtime_state.setdefault("context_snippet", segment[:160])
                runtime_state.setdefault("segment_digest", digest)
                runtime_state["training_active"] = True
                metrics = session.runtime.run_step(runtime_state)
                session._adaptive.track_runtime(metrics)
                gating = session._adaptive.last_gating_summary
                global_step = session.proto_lm.step
                if args.log_every and (step == 1 or step % args.log_every == 0):
                    avg_loss = fmean(losses) if losses else float("nan")
                    print(
                        "  step {local:>7} | global {global_step:>7} | loss {loss:8.4f}"
                        " | avg {avg:8.4f} | seg_len {seg_len:>5} | digest {digest}"
                        " | gate {gate} | novelty {novelty:.2f} | uncertainty {uncertainty:.2f} | drag {drag:.2f}".format(
                            local=step,
                            global_step=global_step,
                            loss=loss,
                            avg=avg_loss,
                            seg_len=len(segment),
                            digest=digest,
                            gate=_render_gate(gating),
                            novelty=float(readings.get("novelty", 0.0)),
                            uncertainty=float(readings.get("uncertainty", 0.0)),
                            drag=float(readings.get("drag", 0.0)),
                        )
                    )
                if args.checkpoint_interval and global_step % args.checkpoint_interval == 0:
                    checkpoint_path = session.proto_lm.save_checkpoint(
                        str(args.checkpoint),
                        reason="interval",
                        metadata={"source": "run_session", "step": global_step},
                        tags=["interactive"],
                    )
                    print(f"  ✔ checkpoint saved to {checkpoint_path}")
                meta_snapshot = session.runtime.meta_state.snapshot()
                memory_snapshot = session.runtime.memory.as_runtime_mapping()
        if stop_event.is_set():
            break
        if losses:
            epoch_loss = fmean(losses)
            print(
                "Epoch {epoch} stats: mean {mean:.4f} | median {med:.4f} | min {minv:.4f} | max {maxv:.4f}"
                " | updates {count} | accepted {accepted} | rejected {rejected}".format(
                    epoch=epoch_index,
                    mean=epoch_loss,
                    med=fmean(sorted([losses[len(losses)//2]])) if losses else float("nan"),
                    minv=min(losses),
                    maxv=max(losses),
                    count=len(losses),
                    accepted=accepted,
                    rejected=rejected,
                )
            )
            checkpoint_path = session.proto_lm.save_checkpoint(
                str(args.checkpoint),
                reason="epoch",
                metadata={"source": "run_session", "epoch": epoch_index},
                tags=["interactive"],
            )
            print(f"  ✔ checkpoint saved to {checkpoint_path}")
        else:
            print("Epoch completed with no accepted segments.")
    print("Training worker finished.")


def telemetry_printer(event: TelemetryEvent) -> None:
    if event.type == "training/step":
        for line in render_training_event(event):
            print(line)
    elif event.type == "ingestion/chunk":
        print(render_ingestion_event(event))
    elif event.type == "training/snapshot":
        payload = event.payload
        if isinstance(payload, Mapping):
            step = payload.get("step", "?")
            snapshot = payload.get("snapshot", {})
            growth = payload.get("growth_events", [])
            print(f"[telemetry] snapshot step={step} loss={snapshot.get('loss')} growth={growth}")


def chat_loop(
    *,
    session: ConversationSession,
    lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    print("\nEnter text to chat. Commands: /scratch, /stoptrain, /quit")
    while not stop_event.is_set():
        try:
            user_text = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            stop_event.set()
            break
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            stop_event.set()
            break
        if user_text == "/scratch":
            with lock:
                scratch = list(session.runtime.scratchpad.current_trace)
            if scratch:
                print("--- Scratchpad ---")
                for line in scratch:
                    print(line)
                print("-------------------")
            else:
                print("(scratchpad empty)")
            continue
        if user_text == "/stoptrain":
            stop_event.set()
            break
        with lock:
            metrics = session.process_user_input(user_text)
            snapshot = session.generate_response()
            session._adaptive.track_runtime(metrics)
            gating = session._adaptive.last_gating_summary
        print(f"assistant: {snapshot.response}")
        if gating is not None:
            decision = getattr(gating, "truth_decision", None)
            if decision is not None:
                print(
                    f"  [gate] decision={decision.decision} score={decision.combined_score:.3f}"
                )
    print("Chat loop exiting.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified SalienceOS runner")
    parser.add_argument("--corpus", type=str, default=None, help="Optional text corpus to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to run (0 = infinite)")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("storage/proto_lm/tinystories.pt"))
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--freeze-learning", action="store_true", help="Disable online learning")
    parser.add_argument("--auto-save-path", type=str, default=None)
    parser.add_argument("--no-training", action="store_true")
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()

    session = create_session(args)
    lock = threading.Lock()
    stop_event = threading.Event()

    unsubscribe = None
    if args.telemetry:
        unsubscribe = BUS.subscribe(telemetry_printer)

    training_thread: Optional[threading.Thread] = None
    if args.corpus and not args.no_training:
        training_thread = threading.Thread(
            target=training_worker,
            kwargs={
                "session": session,
                "args": args,
                "lock": lock,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        training_thread.start()

    dashboard_thread: Optional[threading.Thread] = None
    if args.dashboard:
        dashboard_thread = threading.Thread(
            target=run_dashboard_from_session,
            kwargs={
                "session": session,
                "host": args.host,
                "port": args.port,
            },
            daemon=True,
        )
        dashboard_thread.start()
        print(f"Dashboard available at http://{args.host}:{args.port}")

    try:
        if not args.no_chat:
            chat_loop(session=session, lock=lock, stop_event=stop_event)
        else:
            while not stop_event.is_set():
                time.sleep(1.0)
    finally:
        stop_event.set()
        if training_thread is not None:
            training_thread.join(timeout=5.0)
        if dashboard_thread is not None:
            dashboard_thread.join(timeout=5.0)
        if unsubscribe is not None:
            unsubscribe()
        session.proto_lm.save_checkpoint(
            str(args.checkpoint),
            reason="shutdown",
            metadata={"source": "run_session"},
            tags=["interactive", "shutdown"],
        )
        print("Runner shutting down. Final checkpoint written.")


if __name__ == "__main__":
    main()
