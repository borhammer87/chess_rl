# TRAINING STATUS

## Objective

Build a complete and robust DQN training pipeline before attempting
large-scale training or self-play.

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

### Learning

- [x] Policy network
- [x] Target network
- [x] Periodic target synchronization
- [x] Mini-batch training
- [x] Epsilon decay after successful training updates

---

## Not implemented yet

- [ ] Training metrics
- [ ] Loss history
- [ ] Reward history
- [ ] Epsilon history
- [ ] Model checkpoints
- [ ] Evaluation matches
- [ ] TensorBoard support (optional)

---

## Current limitations

- Only White is controlled by the DQN agent.
- Black is played by RandomAgent.
- The objective is validating the complete training pipeline before
  introducing stronger opponents or self-play.

---

## Next milestone

Implement training metrics to observe learning progress before running
long training sessions.