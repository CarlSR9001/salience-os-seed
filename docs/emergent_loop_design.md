# Emergent Language Loop – SalienceOS Seed Concept

Goal: bootstrap communication without pretraining by exposing the runtime to raw
text files and conversational exchanges. The system must **discover** structure
through repeated salience-driven prediction/verification cycles.

## High-Level Architecture
- **Corpus ingestion (`ingestion/`):** Stream raw files. Build token statistics,
  n-gram counts, and initial sensor baselines (entropy, novelty).
- **Symbol grounding:** Start with byte-pair-ish segmentation. Use salience
  sensors (NEW, AIM, DRAG) to prioritise candidate tokens/subwords.
- **Emergent language model (`proto_lm/`):**
  - Linear SASS core with random init.
  - Online training loop using cross-entropy on next-token prediction.
  - Memory tables track hypotheses about symbol meanings.
  - Verification uses simple mutual information checks and n-gram predictability.
- **Conversational interface (`conversation/`):**
  - Loops user utterances into structured memory.
  - Runtime updates controller based on response success (bandit reward = reduced
    surprisal vs. prior message).

## Learning Dynamics
1. **Read:** ingest chunk, run baseline SASS pass.
2. **Sense:** update sensors: NEW detects unseen n-grams; uncertainty captures
   prediction failures.
3. **Decide:** controller can choose:
   - `MEMORY_OP` to store new facts/words.
   - `SASS` to update internal state via gradient step (online training).
   - `VERIFY` to run consistency checks.
4. **Compute:** training step only triggers when `uncertainty × NEW` above
   threshold (prevent overfitting). Hyper-adapter allows micro specialisation.
5. **Self-awareness:** meta-state tracks confidence in word usage.
6. **Idea loop:** propose new symbols or grammar hypotheses when ROI high.

## Conversation Flow
- User message -> `ConversationSession` encodes into token sequence (current
  vocabulary). Unknown words appended as new tokens.
- Runtime responds by sampling from proto LM; detection of comprehension uses
  DRAG (if repeated attempts fail, escalate to idea loop to request clarification).
- Replay buffer stores message/response pairs to refine predictions.

## Files to add
- `ingestion/reader.py`: stream files, update frequency tables.
- `proto_lm/trainer.py`: online gradient descent with SASS core.
- `conversation/session.py`: handles dialogue state, interface to runtime.
- `conversation/cli.py`: simple chat loop hooking into runtime driver.

## Next steps
1. Implement ingestion + stats.
2. Wire proto LM trainer to runtime (online updates).
3. Build conversational CLI that interops with state generator/driver.
4. Create smoke tests verifying token table growth + decreasing surprisal.
