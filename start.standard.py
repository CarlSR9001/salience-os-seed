"""Launch TinyStories multi-epoch training with preset defaults.

Run this script (``python start.standard.py``) to stream the TinyStories
corpus through the proto language model overnight. It wraps
``salience_os_seed.training.run_corpus`` with configuration tuned for the
``standard/TinyStories-*.txt`` data prepared earlier.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from salience_os_seed.training.run_corpus import train_corpus


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def main() -> None:
    corpus_path = resolve_path(Path("standard/TinyStories-train.txt"))
    checkpoint_path = resolve_path(Path("storage/proto_lm/tinystories.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    args = SimpleNamespace(
        corpus=corpus_path,
        epochs=0,
        chunk_size=2048,
        shuffle_buffer=64,
        seed=13,
        patience=4,
        min_delta=0.05,
        log_every=200,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10000,
        resume=True,
    )

    train_corpus(args)


if __name__ == "__main__":
    main()
