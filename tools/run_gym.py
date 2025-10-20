"""Quick launcher for the chain-of-thought curriculum gym."""

from __future__ import annotations

import argparse
from typing import Iterable

from salience_os_seed.conversation.session import ConversationSession
from salience_os_seed.training.cot_curriculum.loader import CurriculumExample, iter_examples


def run_curriculum(path: str, *, limit: int | None = None) -> None:
    session = ConversationSession()
    examples: Iterable[CurriculumExample] = iter_examples(path)
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
        default="training/cot_curriculum",
        help="Root directory of the curriculum (default: training/cot_curriculum)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples to process",
    )
    args = parser.parse_args()
    run_curriculum(args.path, limit=args.limit)


if __name__ == "__main__":
    main()
