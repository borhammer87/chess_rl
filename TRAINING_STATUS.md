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
- [x] Epsilon decay once per episode when replay training occurred
- [x] Explicit `-0.1` truncation penalty
- [x] Terminal Bellman treatment for truncated replay transitions

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
## Truncation reward

Chess termination and training-horizon termination are intentionally kept
separate.

When a game reaches `max_agent_steps` without a real chess terminal state:

- the environment remains non-terminal,
- the episode is reported as truncated,
- the final replay reward is `-0.1`,
- the final replay transition is stored with `done=True`,
- and `next_legal_actions` is empty.

Treating the replay transition as terminal prevents Bellman bootstrapping
beyond a state for which training deliberately stops generating experience.

The `-0.1` penalty is an initial experimental value and may be adjusted in
future controlled experiments.


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

Validate the truncation penalty with real training data.

The current experiment changes only the artificial-horizon reward:

- normal win: `+1`
- normal loss: `-1`
- normal draw: `0`
- truncation: `-0.1`

Compare the resulting truncation rate, balanced RandomAgent evaluation, and
greedy diagnostic PGN with previous runs.

Do not add material-based shaping or a per-move penalty until this isolated
change has been evaluated.

If the truncation penalty is useful but too weak, later experiments may test
stronger fixed values such as `-0.2` or `-0.3`.

Do not automatically change the penalty during a single training run because
that would mix different reward semantics in the same replay buffer.

Champion-vs-challenger evaluation and promotion criteria remain future work.