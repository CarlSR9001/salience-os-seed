import json
import tempfile

from salience_os_seed.conversation.session import ConversationConfig, ConversationSession


def test_conversation_autosave_cycle(tmp_path):
    auto_path = tmp_path / "state.json"
    config = ConversationConfig(response_tokens=8, auto_save_path=str(auto_path))
    session = ConversationSession(config=config)

    session.process_user_input("hello there")
    snapshot = session.generate_response()
    assert snapshot.response
    assert auto_path.exists()

    # load state into new session
    session2 = ConversationSession(config=config)
    assert list(session2.history)
    saved = json.loads(auto_path.read_text(encoding="utf-8"))
    assert "history" in saved
