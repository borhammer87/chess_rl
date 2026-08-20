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
- [ ] Periodic evaluation during training
- [ ] Best-checkpoint selection

### Persistence

- [x] Agent state serialization
- [x] Replay-buffer state serialization
- [x] Combined training checkpoints
- [x] Periodic `latest.pt` saves
- [x] Automatic loading of `latest.pt`

---

## Current limitations

- Only White is controlled by the DQN agent.
- Black is always played by RandomAgent.
- Self-play is not implemented.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not maintain a global lifetime episode counter.

---

## Next milestone

Integrate evaluation into the training workflow so model performance can
be measured during training and used to identify the best checkpoint.