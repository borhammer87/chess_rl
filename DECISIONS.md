# DECISIONS.md

## Purpose

This document records the main architectural and technical decisions of the Chess Reinforcement Learning project.

The goal is to preserve the reasoning behind each decision, avoid reopening already-settled discussions, and make future refactors easier to evaluate.

---

## D-001 — Use a CNN-based DQN

### Status

Accepted

### Decision

The project uses a convolutional neural network as a Deep Q-Network.

### Reason

The board is represented as spatial data, so a CNN is a natural first model for learning local and positional patterns.

The main goal of the project is to understand Deep Q-Learning clearly before introducing more advanced systems.

### Consequences

- The model predicts one Q-value per action.
- The architecture remains simpler than AlphaZero-style systems.
- No MCTS, policy/value dual heads, transformers, or residual towers are used at this stage.

---

## D-002 — Board representation uses 12 channels

### Status

Accepted

### Decision

The board encoder returns a PyTorch tensor with shape:

```python
(12, 8, 8)
```

The channels represent:

- 6 white piece types
- 6 black piece types

The format is channels-first.

### Reason

This layout is directly compatible with PyTorch `Conv2d`.

### Consequences

- The CNN receives batched inputs with shape `(batch_size, 12, 8, 8)`.
- Castling rights, en passant information, move counters, and repetition state are not represented yet.
- Two positions with identical piece placement but different auxiliary chess state may currently receive the same encoding.

---

## D-003 — Fixed action space of 4096 actions

### Status

Accepted

### Decision

Actions are encoded using:

```python
action = from_square * 64 + to_square
```

The total action space is:

```text
64 × 64 = 4096
```

### Reason

A neural network requires a fixed output size.

The 4096-action encoding is simple, easy to test, and sufficient for building the first complete DQN training pipeline.

### Consequences

- Most output actions are illegal in any given position.
- Legal action masking is required before action selection.
- The encoding does not distinguish promotion piece types.
- The model cannot directly learn the difference between promotion to queen, rook, bishop, or knight.

---

## D-004 — Automatically prefer queen promotion

### Status

Accepted for the initial DQN version

### Decision

When multiple legal promotion moves share the same encoded action, `decode_legal_action()` returns the queen promotion.

Example:

```text
g7g8q
g7g8r
g7g8b
g7g8n
```

All four moves share the same origin and destination, and therefore the same action index.

The decoder selects:

```text
g7g8q
```

### Reason

Supporting underpromotions would require expanding or redesigning the action space, the network output, the decoder, the masking logic, and related tests.

That redesign would distract from the current goal: completing and understanding the DQN pipeline.

Queen promotion is also the strongest and most common promotion choice in ordinary positions.

### Consequences

- The agent cannot choose a rook, bishop, or knight promotion.
- It may fail in rare positions where underpromotion is required for mate, avoiding stalemate, or tactical reasons.
- This limitation must be reconsidered before claiming full chess move coverage.

### Future alternatives

Possible future redesigns include:

1. Add explicit promotion actions to the current action space.
2. Use an action representation similar to AlphaZero's move planes.
3. Represent actions as `(from_square, to_square, promotion_type)`.

No redesign is planned until the basic DQN training pipeline works.

---

## D-005 — Apply legal action masking after inference

### Status

Accepted

### Decision

The CNN always predicts 4096 Q-values.

Illegal actions are masked after inference and before greedy action selection.

Conceptually:

```text
CNN
→ 4096 Q-values
→ illegal actions receive a very negative value
→ argmax selects only a legal action
```

### Reason

The set of legal chess moves changes in every position, while the neural network output size must remain fixed.

### Consequences

- The network still computes values for illegal actions.
- Python-chess remains responsible for determining move legality.
- The greedy branch can only select legal moves after masking.
- The exploration branch must sample directly from the list of legal moves.

---

## D-006 — Exploration samples only legal moves

### Status

Accepted

### Decision

During epsilon-greedy exploration, the agent selects a random move from the legal move list and encodes it.

It does not use:

```python
random.randint(0, 4095)
```

### Reason

Randomly selecting from all 4096 actions would frequently produce illegal moves and break environment interaction.

Exploration means trying different valid actions, not attempting impossible actions.

### Consequences

- Both exploration and exploitation return legal encoded actions.
- `DQNAgent.select_action()` requires the current legal move list.

---

## D-007 — ChessEnv accepts chess.Move objects

### Status

Accepted

### Decision

`ChessEnv.step()` receives a `chess.Move`, not an encoded integer action.

The training layer performs the conversion:

```text
DQNAgent
→ encoded integer action
→ decode_legal_action()
→ chess.Move
→ ChessEnv.step()
```

### Reason

This keeps responsibilities separated:

- `ChessEnv` understands chess rules.
- `DQNAgent` understands tensors and action indices.
- `train_dqn.py` connects the components.

### Consequences

- The environment does not depend on the neural-network action encoding.
- The training layer acts as the integration layer.

---

## D-008 — ChessEnv returns independent board copies

### Status

Accepted

### Decision

`ChessEnv.get_state()` returns a copy of the internal board.

### Reason

Returning the internal `Board` object would allow external code to modify environment state without calling `step()`.

### Consequences

- Environment state changes only through controlled methods.
- Tests and training code can inspect returned boards safely.

---

## D-009 — ReplayBuffer uses deque

### Status

Accepted

### Decision

The replay buffer stores transitions in a fixed-capacity `collections.deque`.

Each transition contains:

```text
state
action
reward
next_state
done
```

### Reason

A deque provides efficient insertion and automatically removes the oldest transition when capacity is reached.

Random replay reduces temporal correlation between consecutive chess positions and allows experiences to be reused.

### Consequences

- Training samples are drawn randomly from previous interactions.
- Very old transitions are discarded once capacity is reached.
- The buffer does not currently implement prioritized replay.

---

## D-010 — Use policy and target networks

### Status

Accepted

### Decision

The DQN agent maintains:

- `policy_net`
- `target_net`

The policy network is trained continuously.

The target network is synchronized periodically.

### Reason

Using the same changing network for predictions and Bellman targets makes DQN unstable.

A temporarily fixed target network provides more stable learning targets.

### Consequences

- Training must track when to update the target network.
- Only the policy network is optimized directly.

---

## D-011 — Current reward is terminal and from White's perspective

### Status

Accepted temporarily; must be reviewed before full training

### Decision

The current environment reward is:

```text
White win: +1
Black win: -1
Draw: 0
Non-terminal move: 0
```

### Reason

This is a simple and unambiguous first reward model.

### Consequences

- Rewards are sparse and delayed.
- The same agent controlling both colors requires careful perspective handling.
- A black-side transition cannot be interpreted correctly without transforming the reward or the state perspective.
- Reward semantics must be resolved before serious self-play training.

### Required follow-up

Before implementing final self-play, decide whether to:

1. train a single agent from side-to-move perspective;
2. negate rewards for black;
3. normalize board encoding to the current player;
4. initially train only one color against a separate opponent.

---

## D-012 — Build training incrementally

### Status

Accepted

### Decision

The training pipeline is implemented in small, tested stages:

1. one agent-environment step;
2. store one transition;
3. run one episode;
4. sample and train from replay;
5. update the target network;
6. decay epsilon;
7. add checkpoints and metrics.

### Reason

Incremental integration makes failures easier to locate and ensures every component is understood.

### Consequences

- Development is slower per feature but more reliable.
- Every major behavior should have tests before moving forward.

---
## D-013 — Separate checkpoint state ownership from checkpoint scheduling

### Status

Accepted

### Decision

Each stateful component is responsible for describing and restoring its
own state.

`DQNAgent` exposes:

- `state_dict()`
- `load_state_dict()`

Its state includes:

- policy network
- target network
- optimizer
- epsilon

`ReplayBuffer` exposes:

- `state_dict()`
- `load_state_dict()`

Its state includes:

- replay capacity
- stored transitions

The training layer composes both component states into a resumable
training checkpoint.

Checkpoint scheduling belongs to the training workflow.

`main()` provides the checkpoint path and callback, while
`train_against_random()` decides when the callback is triggered.

### Reason

The agent should not need to know that a ReplayBuffer exists, and the
ReplayBuffer should not need to know anything about neural networks or
optimizers.

Separating state ownership from checkpoint scheduling keeps component
responsibilities explicit and allows checkpoint policy to evolve without
changing the underlying components.

### Alternatives considered

1. Store the ReplayBuffer directly inside `DQNAgent`.
2. Let `main()` decide both when and how every checkpoint is saved.
3. Save only neural-network weights.

These alternatives were rejected because they either couple unrelated
components or do not preserve enough state to resume training
practically.

### Consequences

- Training can resume with the previous agent and replay memory.
- Checkpoints are larger because replay transitions are serialized.
- The training workflow can later support policies such as `latest.pt`
  and `best.pt` without changing `DQNAgent`.
- Checkpoints do not yet preserve random-number-generator state or a
  global lifetime episode counter.
  
## Known limitations

The current implementation intentionally keeps the scope focused on a
single-agent DQN training pipeline.

Current limitations include:

- The DQN agent always plays White.
- The opponent is always a RandomAgent.
- Self-play is not implemented yet.


---

## Next decision to resolve

The next major design decision is:

> How should rewards and state perspective be represented when the DQN plays both White and Black?

This must be resolved before implementing meaningful self-play training.

## D-014 — Select the best checkpoint using normalized chess score

### Status

Accepted

### Decision

Periodic evaluation results are converted into a normalized score:

`(wins + 0.5 * draws) / episodes`

A win contributes 1 point, a draw 0.5 points, and a loss or truncated
game 0 points.

The project maintains two checkpoint roles:

- `latest.pt` stores the most recent resumable training state.
- `best.pt` stores the state associated with the highest evaluation score
  observed so far.

`best.pt` is replaced only when a new score is strictly greater than the
previous best score.

The best evaluation score is stored as checkpoint metadata.

### Reason

Training recency and playing strength are different concepts.

The latest model is required to resume training, while the best model
should represent the strongest policy observed according to the current
evaluation benchmark.

Using standard chess scoring gives a simple and understandable model
selection criterion.

Normalizing by all evaluation episodes also prevents truncated games
from artificially improving the score.

Persisting the score inside `best.pt` allows model selection to continue
correctly after restarting the program.

### Alternatives considered

1. Use only the number of wins.
2. Ignore truncated games when calculating the denominator.
3. Replace `best.pt` when the new score is equal to the previous score.
4. Keep the best score only in memory.
5. Store the best score in a separate file.

These alternatives were rejected because they either discard useful draw
information, can reward excessive truncation, overwrite equivalent
checkpoints unnecessarily, or make model selection less robust across
training sessions.

### Consequences

- Evaluation results can now drive checkpoint selection.
- `latest.pt` and `best.pt` have explicitly different purposes.
- Evaluation quality directly affects model-selection quality.
- The current score measures performance only against RandomAgent.
- Future evaluation against stronger or multiple opponents may require
  revisiting the selection criterion.

---

## D-015 — Use agent-perspective rewards with absolute board encoding

### Status

Accepted

### Decision

`ChessEnv` continues to expose canonical rewards from White's
perspective.

The training layer converts those rewards to the color controlled by the
DQN:

- White DQN: keep the environment reward.
- Black DQN: reverse the reward sign.

Positive reward therefore always means a favorable outcome for the DQN.

The board representation remains absolute.

White and Black pieces retain their fixed channels, and the board is not
rotated when the DQN plays Black.

Training alternates the DQN between White and Black episodes.

Periodic model-selection evaluation uses equal numbers of games as each
color.

### Reason

The environment should represent chess outcomes independently from which
participant is currently considered the learning agent.

Reward conversion belongs to the training layer because that layer knows
which color the DQN controls.

Keeping an absolute board representation avoids introducing an
additional transformation while the same network learns both colors.

Balanced evaluation prevents `best.pt` from being selected based on
performance with only one color.

### Alternatives considered

1. Make `ChessEnv` return rewards directly from the active agent's
   perspective.
2. Rotate the board and swap piece channels when the DQN plays Black.
3. Train separate networks for White and Black.
4. Continue training and evaluating only as White.

These alternatives were rejected for the current stage because they
either couple the environment to the training agent, introduce additional
representation complexity, duplicate the model, or fail to prepare the
agent for self-play.

### Consequences

- One DQN can learn from White and Black games.
- Replay memory can contain experiences from both colors with consistent
  reward semantics.
- The network must learn color-dependent behavior from an absolute board
  representation.
- Future experiments may revisit agent-relative board normalization if
  learning quality suggests it is beneficial.
- Self-play can now be designed without first solving basic color support.