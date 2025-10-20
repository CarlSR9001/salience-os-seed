"""Quick smoke test for conversation pipeline after salience fixes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from salience_os_seed.conversation.session import ConversationConfig, ConversationSession


def main() -> None:
    config = ConversationConfig(learning_enabled=False)
    session = ConversationSession(config=config)

    prompts = [
        "Hello there!",
        "Can you summarise the state of the runtime?",
        "What should we debug next?",
    ]
    for idx, prompt in enumerate(prompts, start=1):
        session.process_user_input(prompt)
        snapshot = session.generate_response()
        print(f"turn={idx} user={prompt!r}")
        print(f"  response={snapshot.response!r}")
        metrics = snapshot.metrics
        decision = metrics.decision
        print(
            "  runtime_step=", metrics.step,
            "operator=", decision.action.operator.name,
            "score=", decision.score,
        )
        print(f"  meta_report={snapshot.meta_report!r}")
        print("  todos=", list(snapshot.todos))
        if session._last_gating_summary is not None:
            gating = session._last_gating_summary
            print("  gating_truth=", gating.truth_decision.decision)
            print("  truth_star=", f"{gating.truth_star:.3f}")
            print("  combined_score=", f"{gating.combined_score:.3f}")
            if gating.axiom_violations:
                print("  axiom_violations=", [violation.axiom_id for violation in gating.axiom_violations])
            if gating.elegance_accept is not None:
                print("  elegance_accept=", gating.elegance_accept)
            if gating.elegance_score is not None:
                print("  elegance_score=", f"{gating.elegance_score:.3f}")
        print("  budget_left=", session.runtime.budget_left)
        print("  memory_todos=", [todo["text"] for todo in session.runtime.memory.as_runtime_mapping()["todos"]])
        print("---")


if __name__ == "__main__":
    main()
