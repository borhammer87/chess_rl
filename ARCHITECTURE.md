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

`run_dqn_vs_random_episode()` supports the DQN playing either White or
Black.

When the DQN plays Black, RandomAgent performs the opening White move
before the first DQN decision.

Replay transitions store rewards from the DQN's perspective.

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

## Training workflow

`train_against_random()` coordinates multi-episode training.

It supports:

- fixed-color training
- alternating-color training

The current `main()` configuration alternates:

White
→ Black
→ White
→ Black
→ ...

The same agent and replay buffer are reused across all episodes.

## Evaluation

`evaluate_against_random()` evaluates one selected DQN color.

Results are interpreted from the DQN's perspective.

`evaluate_against_random_both_colors()` evaluates equally as White and
Black and combines the results.

The current periodic evaluation uses 10 games per color.

During evaluation:

- epsilon is temporarily set to zero
- training is disabled
- the training replay buffer is not modified
- wins, draws, losses, and truncations are collected
- epsilon is restored afterwards

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