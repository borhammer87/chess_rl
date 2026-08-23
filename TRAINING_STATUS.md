# TRAINING STATUS

## Objective

Build a robust DQN training and evaluation workflow before attempting
self-play or large-scale experiments.

---

## Current pipeline

### Environment

- [x] Chess environment
- [x] Board encoder
- [x] Action encoder
- [x] Legal action masking

### Experience generation

- [x] DQN plays White
- [x] RandomAgent plays Black
- [x] Complete episodes
- [x] Reward generation
- [x] Transition storage

### Replay memory

- [x] ReplayBuffer
- [x] Random replay sampling
- [x] Minimum replay size before training
- [x] Replay-buffer state persistence

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

### Code organization

- [x] Checkpoint persistence separated into `checkpoint.py`
- [x] Result data structures separated into `results.py`
- [x] Episode operations separated into `episodes.py`
- [x] Tests separated according to module responsibility

---

## Model selection

Evaluation performance is normalized using:

`(wins + 0.5 * draws) / episodes`

A win is worth 1 point, a draw 0.5 points, and a loss or truncated game
0 points.

`latest.pt` represents the most recent resumable training state.

`best.pt` represents the highest evaluation score observed so far.

A new evaluation replaces `best.pt` only when its score is strictly
greater than the stored best score.

---

## Current limitations

- Only White is controlled by the DQN agent.
- Black is always played by RandomAgent.
- Self-play is not implemented.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not maintain a global lifetime episode counter.
- Evaluation currently measures performance only against RandomAgent.
- Evaluation scores may vary because evaluation uses a finite sample of
  games.

---

## Next milestone

Decide how training should support the DQN playing both White and Black.

Before implementing self-play, define how rewards and board state should
be represented from the agent's perspective.