# TRAINING STATUS

## Objective

Build a complete DQN training and evaluation pipeline before attempting
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

### Evaluation

- [x] Greedy policy evaluation
- [x] Wins / Draws / Losses
- [x] Truncated games summary

---

## Not implemented yet

- [ ] Model checkpoints
- [ ] Loading trained models
- [ ] Alternate White / Black
- [ ] Self-play
- [ ] TensorBoard integration (optional)

---

## Current limitations

- Only White is controlled by the DQN agent.
- Black is always played by RandomAgent.
- Models cannot yet be saved or restored.

---

## Next milestone

Implement model checkpointing so training can be resumed and evaluated
across multiple sessions.