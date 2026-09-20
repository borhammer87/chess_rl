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

- [x] Design self-play architecture
- [x] Generate self-play episodes
- [x] Define opponent-network update policy
- [x] Integrate self-play into the main training workflow
- [ ] Evaluate self-play agents
- [ ] Stronger opponent benchmarks

## Phase 6 — Learning-signal validation

- [x] Inspect real integrated self-play runs
- [x] Diagnose high truncation rate
- [x] Distinguish claimable draws from genuine truncations
- [x] Implement terminal replay penalty for artificial truncation
- [x] Validate initial `-0.1` truncation penalty in real training
- [x] Determine that truncation penalty alone is insufficient
- [x] Add material-based reward shaping
- [x] Separate chess outcomes from shaped training reward
- [x] Validate combined material shaping and truncation penalty in real training
- [x] Diagnose replay TD-error distribution
- [x] Compare standard DQN and Double-DQN targets diagnostically
- [x] Determine that Double DQN would not change the analysed targets
- [x] Implement Prioritized Experience Replay
- [x] Validate PER in real training
- [x] Diagnose legal-action Q-value separation
- [x] Identify small top-action Q gaps during non-progressing greedy play
- [x] Add a small non-terminal step penalty
- [x] Run initial PER + step-penalty experiment
- [ ] Decide whether the next change should target reward design,
      network architecture, or the DQN formulation itself

## Phase 7 — Future work

- [ ] Hyperparameter tuning
- [ ] Long training experiments
- [ ] Improved board representation
- [ ] Version 1.0