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
- [x] Prioritized Experience Replay
- [x] Priority-based sampling with replacement
- [x] Importance-sampling weights
- [x] TD-error-based priority updates
- [x] Replay-priority persistence
- [x] Minimum replay size before training
- [x] Replay-buffer persistence
- [x] Experiences from White and Black games can share replay memory

### Learning

- [x] Policy network
- [x] Target network
- [x] Periodic target synchronization
- [x] Mini-batch training
- [x] Epsilon decay once per episode when replay training occurred
- [x] Explicit `-0.1` truncation penalty
- [x] Terminal Bellman treatment for truncated replay transitions
- [x] Material-based reward shaping from net material change
- [x] Chess-outcome classification independent of shaped reward
- [x] Importance-sampling weighted per-transition loss
- [x] Absolute TD-error reporting for replay-priority updates
- [x] Small `-0.0005` step penalty on non-terminal, non-truncated transitions

### Training metrics

- [x] Episode reward
- [x] Training losses
- [x] Final epsilon
- [x] Replay buffer size
- [x] Training summary
- [x] Console progress reporting
- [x] Claimable-threefold truncation count
- [x] Claimable-fifty-move truncation count
- [x] Truncations without claimable draw

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
- [x] Greedy diagnostic game against RandomAgent
- [x] Diagnostic PGN export

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
## Training reward

The learning reward now combines several signals.

Real chess termination uses the canonical environment reward, converted
to the DQN's perspective:

- win: `+1`
- draw: `0`
- loss: `-1`

Intermediate transitions receive material shaping:

`material_reward = 0.01 * net material-balance change`

Piece values are:

- pawn: 1
- knight: 3
- bishop: 3
- rook: 5
- queen: 9

The material change is measured across the full DQN transition, including
the opponent response.

Ordinary non-terminal, non-truncated learner transitions additionally receive:

`STEP_PENALTY = -0.0005`

This creates a small cumulative cost for prolonged play while remaining much
smaller than the terminal chess reward.

Real terminal transitions do not receive the step penalty.

Artificial truncation adds:

`TRUNCATION_PENALTY = -0.1`

to the final transition.

The final transition of a truncated episode is also stored as terminal for
Bellman learning even though the chess environment itself has not reached
a real terminal state.

The step penalty is not added to the artificially truncated final transition;
the separate `-0.1` truncation penalty applies there instead.

The `0.01` material scale and `-0.1` truncation penalty are experimental
values, not tuned hyperparameters.

## Prioritized replay

Training currently uses Prioritized Experience Replay rather than uniform
replay sampling.

Current experimental values:

- `PER_ALPHA = 0.6`
- `PER_BETA = 0.4`
- `PER_PRIORITY_EPSILON = 1e-6`

New transitions initially receive the current maximum replay priority.

Sampling is performed with replacement.

The DQN loss is calculated per transition, multiplied by normalized
importance-sampling weights, and then averaged.

After training, sampled priorities are updated from absolute TD error plus
the small priority epsilon.

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

## Outcome classification

Chess outcome and training reward are separate concepts.

Win, draw, and loss counts are determined from the actual chess result
stored in `final_info["result"]`, interpreted according to `agent_color`.

`total_reward` is not used to determine the game outcome.

This separation is necessary because material shaping can make the sign of
the accumulated learning reward differ from the actual chess result.

## Next milestone

Evaluate the completed PER + step-penalty experiment before changing the
learning system again.

The current experimental reward is:

- terminal win: `+1`
- terminal draw: `0`
- terminal loss: `-1`
- material change: `0.01 * net material-balance change`
- ordinary non-terminal step: additional `-0.0005`
- artificial truncation: additional `-0.1` instead of the step penalty

Prioritized replay currently uses:

- `alpha = 0.6`
- `beta = 0.4`
- priority epsilon `1e-6`

The first 100-episode run with this configuration still showed a high
truncation rate and weak greedy RandomAgent performance.

Before another training modification, analyse whether the persistent problem
is best addressed through reward design, model architecture, or a more
fundamental change to the learning formulation.