"""Sample text from the synthetic baseline checkpoint after loading it."""
from pathlib import Path

from salience_os_seed.conversation.session import (
    ConversationConfig,
    ConversationSession,
    TrainingConfig,
)


CHECKPOINT = Path("storage/proto_lm/synthetic_baseline.pt")


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"Missing checkpoint at {CHECKPOINT}")
    cfg = ConversationConfig(
        lm=TrainingConfig(checkpoint_path=str(CHECKPOINT)),
        learning_enabled=False,
        archive_checkpoint_on_start=False,
        response_tokens=80,
    )
    session = ConversationSession(config=cfg)
    print(f"Loaded step: {session.proto_lm.step}")
    print(f"Vocab size: {session.proto_lm.vocab.size()}")
    prompt = "Hello, assistant!"
    sample = session.proto_lm.sample(prompt, max_tokens=80)
    print("Prompt:", prompt)
    print("Sample:")
    print(sample)


if __name__ == "__main__":
    main()
