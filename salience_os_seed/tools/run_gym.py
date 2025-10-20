"""Quick launcher for the chain-of-thought curriculum gym."""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
from typing import Iterable

from salience_os_seed.conversation.session import ConversationSession
from salience_os_seed.training.cot_curriculum.loader import CurriculumExample, iter_examples


DEFAULT_CURRICULUM_ROOT = resources.files("salience_os_seed.training").joinpath("cot_curriculum")


def run_curriculum(path: str | Path, *, limit: int | None = None) -> None:
    session = ConversationSession()
    examples: Iterable[CurriculumExample] = iter_examples(str(path))
    processed = 0
    for example in examples:
        session.process_user_input(example.task)
        session.generate_response("\n".join(example.reasoning_trace))
        processed += 1
        if limit is not None and processed >= limit:
            break
    print(f"Processed {processed} curriculum examples from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CoT curriculum gym through the runtime")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_CURRICULUM_ROOT),
        help=(
            "Root directory of the curriculum (default: bundled cot curriculum under "
            "salience_os_seed.training)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples to process",
    )
    args = parser.parse_args()
    run_curriculum(Path(args.path), limit=args.limit)


if __name__ == "__main__":
    main()
