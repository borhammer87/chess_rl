# ROADMAP

## Phase 1 — Core infrastructure

- [x] Chess environment
- [x] Board encoder
- [x] Action encoder
- [x] Legal action masking
- [x] DQN CNN
- [x] Replay Buffer

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

## Phase 4 — Evaluation

- [x] Evaluation against RandomAgent
- [x] Integrate evaluation into the training workflow
- [x] Evaluation scoring
- [x] Best-checkpoint selection
- [ ] Alternate playing White and Black

## Phase 5 — Future work

- [ ] Self-play
- [ ] Stronger opponents
- [ ] Hyperparameter tuning
- [ ] Long training experiments
- [ ] Version 1.0