# ROADMAP

This roadmap distinguishes implemented milestones from validation that is still outstanding.

## Phase 1 — Core chess/RL infrastructure

- [x] Chess environment
- [x] Board encoder
- [x] Action encoder
- [x] Legal action masking/filtering
- [x] Original CNN DQN
- [x] Replay buffer

## Phase 1.1 — State and action completeness

- [x] Encode side to move
- [x] Encode White and Black castling rights
- [x] Encode en-passant target square
- [x] Expand board representation from 12 to 18 channels
- [x] Represent promotions explicitly
- [x] Support queen, rook, bishop and knight promotion
- [x] Expand action space from 4096 to 4272 actions

## Phase 2 — Original DQN training pipeline

- [x] DQN vs RandomAgent
- [x] Replay sampling and Bellman updates
- [x] Multi-episode training
- [x] Target-network synchronization
- [x] Epsilon scheduling after successful episode training
- [x] Training metrics
- [x] `TrainingSummary`

## Phase 3 — Executable training workflow and persistence

- [x] Main training execution
- [x] Console progress reporting
- [x] Periodic checkpoints
- [x] Resume from `latest.pt`
- [x] Replay-buffer persistence
- [x] Checkpoint metadata support

## Phase 4 — Color support and evaluation

- [x] Learner plays White and Black
- [x] Agent-perspective rewards
- [x] Alternating training colors
- [x] Greedy RandomAgent evaluation
- [x] Balanced White/Black evaluation
- [x] Evaluation scoring
- [x] `best.pt` selection
- [x] Greedy diagnostic PGN export

## Phase 5 — Frozen-opponent self-play

- [x] Independent frozen policy opponent
- [x] Generic learner-vs-opponent episode engine
- [x] Multi-episode frozen-opponent training
- [x] Alternating colors in self-play
- [x] Independent target-network and opponent refresh schedules
- [x] Integrate original DQNCNN self-play into `main()`
- [ ] Add stronger/historical opponent evaluation when justified
- [ ] Champion-vs-challenger promotion system if/when a promotion criterion is defined

## Phase 6 — Original-DQN learning diagnostics

- [x] Diagnose high truncation rate
- [x] Record claimable-draw state at truncation
- [x] Keep chess termination semantics after diagnosis
- [x] Add terminal Bellman treatment and `-0.1` penalty for artificial truncation
- [x] Add material-based reward shaping
- [x] Separate chess outcome from shaped reward
- [x] Diagnose replay TD-error distribution
- [x] Compare standard and Double-DQN targets diagnostically
- [x] Implement Prioritized Experience Replay
- [x] Diagnose legal-action Q-value separation
- [x] Add small ordinary step penalty
- [x] Conclude from documented experiments that these changes had not yet established reliable greedy play

## Phase 7 — Alternative State-Action DQN

- [x] Preserve original DQNCNN in parallel
- [x] Implement explicit State-Action network
- [x] Structured action decoding and embeddings
- [x] Shared Q-head
- [x] Reuse state encodings across candidate actions
- [x] Batch selected state-action pairs
- [x] Batch legal-next-action maxima
- [x] Implement `StateActionDQNAgent`
- [x] Support PER weights and TD-error priority updates
- [x] Support agent serialization/checkpoints
- [x] Complete short end-to-end CPU RandomAgent training smoke test
- [x] Run controlled validation beyond smoke-test level
- [x] Add reproducible State-Action CPU probe with controlled initialization, training and evaluation seeds
- [x] Run controlled 100-episode and 500-episode CPU probes
- [ ] Diagnose persistent greedy truncation and the 100-to-500-episode learning plateau
- [ ] Compare greedy performance with original DQNCNN
- [ ] Decide whether State-Action remains parallel or becomes the primary model
- [ ] Integrate State-Action into frozen-opponent self-play only if validation warrants it

## Phase 8 — Workflow/reproducibility improvements

- [ ] Declare runtime/development dependencies in project metadata or an equivalent reproducible environment specification
- [ ] Consider RNG-state persistence if exact resume reproducibility becomes a requirement
- [ ] Consider a lifetime episode counter if training history requires it
- [ ] Add explicit device/CUDA support if future training scale justifies it; current CPU probe cost does not make it an immediate priority

## Phase 9 — Longer-term work

- [ ] Stronger evaluation methodology
- [ ] Hyperparameter tuning
- [ ] Longer controlled training experiments
- [ ] Improved state representation if diagnostics justify it
- [ ] Version 1.0 criteria and release
