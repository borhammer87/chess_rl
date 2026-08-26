# ROADMAP

## Phase 1 — Core infrastructure

- [x] Chess environment
- [x] Board encoder
- [x] Action encoder
- [x] Legal action masking
- [x] DQN CNN
- [x] Replay Buffer

## Phase 1.1 -State and action completeness

- [x] Encode side to move
- [x] Encode White and Black castling rights
- [x] Encode en passant target square
- [x] Expand board representation from 12 to 18 channels
- [x] Add explicit promotion actions
- [x] Support queen promotion
- [x] Support rook underpromotion
- [x] Support bishop underpromotion
- [x] Support knight underpromotion
- [x] Expand action space from 4096 to 4272 actions

## Phase 2 — Training pipeline

- [x] DQN vs RandomAgent
- [x] Replay sampling
- [x] Multi-episode training
- [x] Target network synchronization
- [x] Epsilon scheduling
- [x] Training metrics collection
- [x] Training summary generation

## Phase 3 — Training workflow

- [x] Training execution
- [x] Progress reporting
- [x] Periodic checkpoints
- [x] Load previous training state
- [x] Replay-buffer persistence

## Phase 4 — Evaluation and balanced colors

- [x] Evaluation against RandomAgent
- [x] Integrate evaluation into the training workflow
- [x] Evaluation scoring
- [x] Best-checkpoint selection
- [x] Agent-perspective rewards
- [x] DQN plays White
- [x] DQN plays Black
- [x] Alternate White and Black during training
- [x] Balanced evaluation as White and Black

## Phase 5 — Self-play

- [ ] Design self-play architecture
- [ ] Generate self-play episodes
- [ ] Define opponent-network update policy
- [ ] Evaluate self-play agents
- [ ] Stronger opponent benchmarks

## Phase 6 — Future work

- [ ] Hyperparameter tuning
- [ ] Long training experiments
- [ ] Improved board representation
- [ ] Version 1.0