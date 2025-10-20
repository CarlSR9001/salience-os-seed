# Emergent Conversation CLI

Launch the unsupervised chat loop from project root:
```powershell
python -m salience_os_seed.conversation.cli --autosave storage/convo_state.json
```

## Options
- `--corpus PATH` — optional directory of `.txt`, `.md`, or `.jsonl` files to bootstrap vocabulary.
- `--autosave PATH` — persist emergent vocabulary, RNG state, memory, and dialogue history after each exchange.
- `--response-tokens N` — clamp generated response length (default 48).
- `--steps N` — exit automatically after N user exchanges.
- `--quiet` — suppress meta-state telemetry for a minimal chat view.

## Workflow
1. The session ingests your message via `ConversationSession.process_user_input()`.
2. `ProtoLanguageModel.training_step()` performs an incremental update using only seen text.
3. `SalienceRuntime.run_step()` emits salience metrics influencing memory ops and verification.
4. `generate_response()` samples from the emergent vocabulary and records the result in structured memory.
5. Autosave (if enabled) writes `conversation_state.json` so future sessions resume vocabulary, RNG state, and todos.

## Resuming a Session
```powershell
python -m salience_os_seed.conversation.cli --autosave storage/convo_state.json
```
The CLI loads saved state automatically if the file exists.
