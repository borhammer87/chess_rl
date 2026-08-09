# TRAINING STATUS

## Objective

Build a complete DQN training pipeline before attempting long training
sessions or self-play.

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

### Metrics

- [x] Episode reward
- [x] Training losses
- [x] Final epsilon
- [x] Replay buffer size
- [x] Training summary

---

## Not implemented yet

- [ ] Training execution from main()
- [ ] Progress reporting
- [ ] Model checkpoints
- [ ] Evaluation matches
- [ ] TensorBoard integration (optional)

---

## Current limitations

- Only White is controlled by the DQN agent.
- Black is always played by RandomAgent.
- No model persistence.
- No evaluation workflow yet.

---

## Next milestone

Run multi-episode training from the main program and report aggregated
training metrics after each execution.