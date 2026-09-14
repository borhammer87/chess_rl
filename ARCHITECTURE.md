# ARCHITECTURE

## Core interaction flow

ChessEnv
→ BoardEncoder (18 × 8 × 8)
→ DQNCNN
→ 4272 Q-values
→ Legal Mask
→ decode_legal_action
→ ChessEnv.step
→ reward_for_color
→ ReplayBuffer
→ train_step

## State representation

The board encoder returns a tensor with shape:

`(18, 8, 8)`

Channels:

- 0–5: White pieces
- 6–11: Black pieces
- 12: White kingside castling right
- 13: White queenside castling right
- 14: Black kingside castling right
- 15: Black queenside castling right
- 16: En passant target square
- 17: Side to move

The representation remains absolute: the board is not rotated when the
DQN plays Black.

## Action representation

The DQN uses a fixed action space of 4272 actions.

- Actions 0–4095 preserve the original `from_square * 64 + to_square`
  encoding for non-promotion moves.
- Actions 4096–4271 represent explicit promotion actions.
- Queen, rook, bishop, and knight promotions have distinct action indices.

The agent can therefore learn underpromotions rather than implicitly
defaulting every promotion to a queen.

## Training package

Training responsibilities are separated by concern.

### `results.py`

Defines the data structures returned by training and evaluation:

- `StepResult`
- `EpisodeResult`
- `VsRandomEpisodeResult`
- `TrainingSummary`
- `EvaluationSummary`

### `episodes.py`

Contains lower-level interaction and learning operations:

- `reward_for_color()`
- `run_single_step()`
- `run_and_store_step()`
- `train_from_replay()`
- `run_episode()`
- `run_dqn_vs_random_episode()`
- `run_dqn_vs_opponent_episode()`

`run_dqn_vs_random_episode()` supports the DQN playing either White or
Black.

When the DQN plays Black, RandomAgent performs the opening White move
before the first DQN decision.

Replay transitions store rewards from the DQN's perspective.

Artificial training truncation is handled separately from real chess
termination.

If the episode reaches `max_agent_steps` while `ChessEnv` is still
non-terminal:

- the episode remains classified as truncated,
- the final learner replay transition receives `TRUNCATION_PENALTY = -0.1`,
- that replay transition is stored with `done=True`,
- and its `next_legal_actions` list is empty.

This terminal flag belongs to the DQN learning transition rather than to the
chess environment. It prevents Bellman bootstrapping beyond the artificial
training horizon.

`run_dqn_vs_opponent_episode()` contains the common DQN-versus-opponent
episode logic.

Opponent-specific behavior is injected through an
`OpponentMoveSelector`, allowing RandomAgent and frozen DQN opponents to
reuse the same episode engine.

### `train_dqn.py`

Coordinates complete training runs.

Its responsibilities include:

- Multi-episode training
- Alternating DQN color
- Progress callbacks
- Target-network synchronization
- Checkpoint scheduling
- Evaluation scheduling
- Balanced White/Black evaluation
- Training summaries
- Best-model selection
- Main program execution
- Truncation diagnostics
- Greedy diagnostic-game generation
- PGN diagnostic export

### `checkpoint.py`

Composes and restores resumable training checkpoints.

## Color model

`ChessEnv` remains color-neutral from the training architecture's point
of view and returns canonical White-perspective rewards.

The training layer converts rewards according to the color controlled by
the DQN.

The same DQN network is used for White and Black.

Board encoding remains absolute:

- White piece channels remain White.
- Black piece channels remain Black.
- The board is not rotated when the DQN plays Black.

## Terminal-state semantics

The architecture distinguishes two kinds of termination.

### Chess termination

`ChessEnv` determines whether the chess game itself has ended.

Examples include:

- checkmate,
- stalemate,
- insufficient material,
- and other automatic game-over conditions recognized by `python-chess`.

### Training-horizon termination

Training additionally limits the number of learner decisions with
`max_agent_steps`.

Reaching this limit does not change `ChessEnv.done`.

Instead, the episode is classified as truncated and the final replay
transition is treated as terminal for DQN learning.

This separation preserves correct chess semantics while defining a finite
learning horizon for the Bellman target.

## Training workflow

The project provides two multi-episode training workflows.

### Training against RandomAgent

`train_against_random()` coordinates multi-episode DQN training against
`RandomAgent`.

It supports:

- fixed-color training
- alternating-color training

This workflow remains available and tested, but it is no longer the
training workflow used by `main()`.

### Frozen-opponent self-play

`train_against_frozen()` coordinates multi-episode training against a
frozen DQN opponent.

The frozen opponent is an independent copy of the learner's `policy_net`.

It:

- starts with the learner policy weights,
- runs in evaluation mode,
- has gradients disabled,
- selects legal moves greedily,
- can be periodically refreshed from the current learner policy.

The current `main()` configuration uses this frozen-opponent self-play
workflow and alternates the learner color:

White
→ Black
→ White
→ Black
→ ...

The same learner and replay buffer are reused across all episodes.

When resuming training, `main()` loads `latest.pt` before creating the
frozen opponent. This ensures that the opponent is initialized from the
restored learner policy rather than from newly initialized weights.

The current `main()` configuration uses:

- target-network synchronization every 10 episodes,
- frozen-opponent refresh every 25 episodes,
- training checkpointing every 25 episodes,
- RandomAgent evaluation every 25 episodes.

Frozen-opponent synchronization and target-network synchronization are
independent mechanisms.

`target_net` stabilizes Bellman targets during DQN learning.

The frozen opponent provides a temporarily stable adversary during
self-play.

## Evaluation

`evaluate_against_random()` evaluates one selected DQN color.

Results are interpreted from the DQN's perspective.

`evaluate_against_random_both_colors()` evaluates equally as White and
Black and combines the results.

The current periodic evaluation uses 10 games per color.

RandomAgent is the current provisional stable evaluation benchmark.

It is used for periodic evaluation and best-model selection, while the
main training workflow uses the frozen DQN opponent described above.

During evaluation:

- epsilon is temporarily set to zero
- training is disabled
- the training replay buffer is not modified
- wins, draws, losses, and truncations are collected
- epsilon is restored afterwards

After the main training run, `main()` also plays one additional greedy
diagnostic game against RandomAgent.

This game:

- temporarily uses epsilon `0`,
- does not train the agent,
- does not modify the training replay buffer,
- is saved as `checkpoints/evaluation_game.pgn`.

The diagnostic PGN is intended for qualitative inspection of the learned
policy and is not used for checkpoint scoring.

## Persistence

### DQNAgent

Owns:

- Policy network
- Target network
- Optimizer
- Epsilon

### ReplayBuffer

Owns:

- Capacity
- Stored transitions

### Training checkpoints

`checkpoint.py` combines both states and optional metadata.

Two checkpoint roles exist:

- `checkpoints/latest.pt` — latest resumable training state
- `checkpoints/best.pt` — highest balanced evaluation score

The model-selection score is:

`(wins + 0.5 * draws) / episodes`

Losses and truncated games contribute zero points.

`best.pt` is replaced only when a new score is strictly greater than the
stored score.

### `self_play.py`

Contains the frozen-opponent self-play infrastructure used by the main
training workflow:

- Creation of an independent frozen copy of `policy_net`
- Greedy legal move selection for the frozen opponent
- Frozen-opponent selector adapter
- DQN-versus-frozen episode wrapper
- Multi-episode self-play
- Alternating learner color
- Periodic frozen-opponent synchronization

The frozen opponent and the DQN target network have different roles.

`target_net` stabilizes Bellman targets.

The frozen opponent provides a temporarily stable adversary during
self-play.

Both are copied from `policy_net`, but their update schedules are
independent.