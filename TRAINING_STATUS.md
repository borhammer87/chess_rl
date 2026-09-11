# TRAINING STATUS

## Objective

Build and validate a robust DQN self-play training workflow before moving
to larger experiments and more advanced opponent-selection strategies.

---

## Current pipeline

### Environment

- [x] Chess environment
- [x] Board encoder
- [x] Action encoder
- [x] Legal action masking

### Experience generation

- [x] DQN plays White
- [x] DQN plays Black
- [x] RandomAgent can play the opposite color
- [x] Alternate DQN color between training episodes
- [x] Complete episodes
- [x] Reward generation
- [x] Agent-perspective reward conversion
- [x] Transition storage

### Replay memory

- [x] ReplayBuffer
- [x] Random replay sampling
- [x] Minimum replay size before training
- [x] Replay-buffer persistence
- [x] Experiences from White and Black games can share replay memory

### Learning

- [x] Policy network
- [x] Target network
- [x] Periodic target synchronization
- [x] Mini-batch training
- [x] Epsilon decay after successful training updates

### Training metrics

- [x] Episode reward
- [x] Training losses
- [x] Final epsilon
- [x] Replay buffer size
- [x] Training summary
- [x] Console progress reporting

### Evaluation

- [x] Greedy policy evaluation
- [x] White-perspective evaluation
- [x] Black-perspective evaluation
- [x] Balanced evaluation across both colors
- [x] Wins / Draws / Losses
- [x] Truncated games summary
- [x] Periodic evaluation during training
- [x] Normalized evaluation scoring
- [x] Best-checkpoint selection

### Persistence

- [x] Agent state serialization
- [x] Replay-buffer state serialization
- [x] Combined training checkpoints
- [x] Optional checkpoint metadata
- [x] Periodic `latest.pt` saves
- [x] Automatic loading of `latest.pt`
- [x] Persistent `best.pt` evaluation score
- [x] Automatic replacement of `best.pt` after improvement

### Self-play

- [x] Frozen opponent copied from current policy
- [x] Frozen opponent remains independent from the learner
- [x] Frozen opponent uses greedy legal action selection
- [x] Shared generic episode engine
- [x] Multi-episode self-play
- [x] Alternate learner color
- [x] Periodic target-network synchronization
- [x] Periodic frozen-opponent synchronization
- [ ] Self-play evaluation
- [x] Main-program self-play integration

### Code organization

- [x] Checkpoint persistence separated into `checkpoint.py`
- [x] Result data structures separated into `results.py`
- [x] Episode operations separated into `episodes.py`
- [x] Tests separated according to module responsibility

---

## Reward perspective

`ChessEnv` returns canonical White-perspective rewards.

The training layer converts rewards using the DQN's current color.

This guarantees:

`positive reward = good for the DQN`

regardless of whether the DQN is playing White or Black.

---

## Board perspective

The board representation remains absolute.

White and Black pieces always occupy their fixed encoder channels.

The board is not rotated or color-normalized for the DQN.

This decision keeps the current representation simple while allowing the
same network to learn policies for both colors.

---

## Model selection

Periodic evaluation uses an equal number of games as White and Black. RandomAgent is currently used as the provisional stable evaluation
benchmark, while training itself uses the frozen DQN opponent.

Evaluation performance is normalized using:

`(wins + 0.5 * draws) / episodes`

`latest.pt` represents the most recent resumable training state.

`best.pt` represents the highest balanced evaluation score observed so
far.

---

## Current limitations

- Board encoding does not include
  repetition state, or move counters.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not maintain a global lifetime episode counter.
- Evaluation currently measures performance only against RandomAgent.

---

## Next milestone

Run the integrated self-play workflow from the main program and inspect
real training behaviour and RandomAgent evaluation results.

The immediate goal is to validate the current training workflow before
introducing more advanced opponent-selection mechanisms.

Champion-vs-challenger evaluation and promotion criteria remain future
work.