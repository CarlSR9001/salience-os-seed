"""Evaluate the synthetic baseline checkpoint on tiny synthetic splits.

Example usage::

    python eval.synthetic_baseline.py --checkpoint storage/proto_lm/synthetic_baseline.pt

The script loads the checkpoint, computes lightweight metrics on the
synthetic validation/test corpora, and emits qualitative conversational samples.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TYPE_CHECKING

import torch


if TYPE_CHECKING:  # pragma: no cover - import-time conveniences only
    from salience_os_seed.conversation.session import ConversationSession


def _ensure_repo_on_path() -> None:
    """Ensure the repository root is importable when running as a script."""

    parent = Path(__file__).resolve().parent.parent
    parent_str = str(parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)


DEFAULT_VAL_PATH = Path("data/local_benchmarks/synthetic_baseline_validation.txt")
DEFAULT_TEST_PATH = Path("data/local_benchmarks/synthetic_baseline_test.txt")


@dataclass
class EvalConfig:
    checkpoint: Path
    device: str = "auto"
    max_samples: int = 2048
    chunk_size: int = 2048
    prompt_prefixes: list[str] | None = None
    temperature: float = 0.8
    max_gen_tokens: int = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a synthetic baseline checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint .pt file (defaults to live synthetic baseline)")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH, help="Validation split path")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH, help="Test split path (optional)")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum validation samples to evaluate")
    parser.add_argument("--chunk-size", type=int, default=512, help="Characters per evaluation chunk")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for qualitative outputs")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling cutoff")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to previously generated tokens (1.0 disables)",
    )
    parser.add_argument(
        "--stop-sequence",
        action="append",
        dest="stop_sequences",
        default=None,
        help="Additional stop sequence for sampling (can be repeated)",
    )
    parser.add_argument("--max-gen-tokens", type=int, default=120, help="Maximum generation length for samples")
    parser.add_argument("--no-copy", action="store_true", help="Load the checkpoint in place without making an eval copy")
    return parser.parse_args()


def load_session(ckpt: Path, device: str) -> ConversationSession:
    _ensure_repo_on_path()
    from salience_os_seed.conversation.session import ConversationConfig, ConversationSession
    from salience_os_seed.proto_lm.trainer import TrainingConfig

    training = TrainingConfig()
    training.checkpoint_path = str(ckpt)
    training.device = device
    config = ConversationConfig(lm=training, learning_enabled=False)
    session = ConversationSession(config=config)
    session.proto_lm.load_checkpoint(str(ckpt))
    session.proto_lm.set_learning_enabled(False)
    return session


def iter_chunks(path: Path, chunk_size: int, max_samples: int) -> Iterator[str]:
    count = 0
    buffer: list[str] = []
    length = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples and count >= max_samples:
                break
            text = line.strip()
            if not text:
                continue
            buffer.append(text)
            length += len(text) + 1
            if length >= chunk_size:
                yield "\n".join(buffer)
                buffer.clear()
                length = 0
                count += 1
    if buffer and (not max_samples or count < max_samples):
        yield "\n".join(buffer)


@torch.no_grad()
def compute_metrics(
    session: ConversationSession,
    corpus: Path,
    *,
    chunk_size: int,
    max_samples: int,
) -> dict[str, float]:
    model = session.proto_lm
    losses: list[float] = []
    total_tokens = 0
    correct_next = 0
    total_next = 0
    for chunk in iter_chunks(corpus, chunk_size, max_samples):
        ids = model.encode(chunk, mutate=False)
        if len(ids) < 2:
            continue
        inputs = torch.tensor(ids[:-1], device=model.device).unsqueeze(0)
        targets = torch.tensor(ids[1:], device=model.device).unsqueeze(0)
        logits = model._forward_logits(inputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = torch.nn.functional.nll_loss(log_probs[:, :-1, :].reshape(-1, log_probs.size(-1)), targets[:, :-1].reshape(-1))
        losses.append(loss.item())
        preds = log_probs.argmax(dim=-1)
        mask = targets[:, :-1]
        correct_next += (preds[:, :-1] == mask).sum().item()
        total_next += mask.numel()
        total_tokens += mask.numel()
    if not losses:
        return {"loss": float("nan"), "perplexity": float("nan"), "next_token_acc": 0.0, "tokens": 0}
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "next_token_acc": correct_next / max(total_next, 1),
        "tokens": total_tokens,
        "samples": len(losses),
    }


def qualitative_samples(
    session: ConversationSession,
    prompts: Iterable[str],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_tokens: int,
    stop_sequences: tuple[str, ...],
) -> list[dict[str, str]]:
    model = session.proto_lm
    samples = []
    for prefix in prompts:
        generated = model.sample(
            prefix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        )
        samples.append({"prompt": prefix, "continuation": generated[len(prefix):]})
    return samples


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        target_ckpt = args.checkpoint
    else:
        live_ckpt = Path("storage/proto_lm/synthetic_baseline.pt")
        if not live_ckpt.exists():
            raise FileNotFoundError(f"Live checkpoint not found: {live_ckpt}")
        if args.no_copy:
            target_ckpt = live_ckpt
        else:
            eval_dir = live_ckpt.parent / "checkpoints_eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            target_ckpt = eval_dir / f"{live_ckpt.stem}-eval-{timestamp}{live_ckpt.suffix or '.pt'}"
            shutil.copy2(live_ckpt, target_ckpt)
            print(f"Copied live checkpoint to {target_ckpt}")

    val_path = args.val
    if not val_path.exists():
        raise FileNotFoundError(f"Validation corpus not found: {val_path}")
    test_path = args.test
    if test_path and not test_path.exists():
        print(f"Warning: test corpus not found at {test_path}; skipping test metrics")
        test_path = None

    session = load_session(target_ckpt, device="auto")
    print(f"Loaded checkpoint step={session.proto_lm.step} from {target_ckpt} on device={session.proto_lm.device}")

    metrics_val = compute_metrics(session, val_path, chunk_size=args.chunk_size, max_samples=args.max_samples)
    print("Validation metrics:")
    print(json.dumps(metrics_val, indent=2))

    metrics_test = None
    if test_path:
        metrics_test = compute_metrics(session, test_path, chunk_size=args.chunk_size, max_samples=args.max_samples)
        print("Test metrics:")
        print(json.dumps(metrics_test, indent=2))

    prompts = [
        "User: Hello! Assistant:",
        "User: What is 5 plus 6? Assistant:",
        "User: Please summarize a sunny weather report. Assistant:",
        "User: Activate the calculator tool for 2 * 3. Assistant:",
    ]
    default_stops = tuple(args.stop_sequences) if args.stop_sequences else ("\n\n",)
    completions = qualitative_samples(
        session,
        prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_gen_tokens,
        stop_sequences=default_stops,
    )
    print("Qualitative samples:")
    for sample in completions:
        print("---")
        print("Prompt:", sample["prompt"])
        print("Continuation:", sample["continuation"])


if __name__ == "__main__":
    main()
