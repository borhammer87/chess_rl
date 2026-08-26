# CURRENT STATE

## Status

The project currently contains a complete DQN training, evaluation, and
checkpoint-selection workflow against RandomAgent.

The DQN can train and evaluate as both White and Black.

Implemented components:

- Chess environment
- Board encoder
- Action encoder
- Legal action masking
- DQN CNN
- Replay Buffer
- DQN vs RandomAgent training
- Multi-episode training
- Replay sampling
- Target network synchronization
- Epsilon decay after successful training updates
- Training metrics collection
- Training summary generation
- Console progress reporting
- Agent-perspective rewards
- Training as White and Black
- Alternating colors between training episodes
- Evaluation against RandomAgent as White and Black
- Balanced two-color evaluation
- Periodic evaluation during training
- Evaluation scoring
- Periodic checkpointing
- Training checkpoint loading
- Replay-buffer persistence
- Best-checkpoint selection
- Extended 18-channel board representation
- Side-to-move encoding
- Castling-rights encoding
- En passant target-square encoding
- Explicit queen, rook, bishop, and knight promotion actions
- 4272-action DQN output space

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

## Training package structure

Training responsibilities are separated into focused modules:

- `results.py` — training and evaluation result data structures.
- `episodes.py` — step, episode, reward-perspective, and replay-training
  operations.
- `checkpoint.py` — training-state persistence and checkpoint metadata.
- `train_dqn.py` — multi-episode workflow, color alternation, evaluation
  scheduling, model selection, summaries, and main program execution.

## Color and reward semantics

`ChessEnv` keeps a canonical White-perspective reward:

- White win: `+1`
- Black win: `-1`
- Draw or unfinished game: `0`

The training layer converts this reward to the DQN agent's perspective.

Therefore:

- Positive reward always represents a good outcome for the DQN.
- Negative reward always represents a bad outcome for the DQN.
- Draws remain neutral.

The board encoder remains absolute:

- Channels 0–5 represent White pieces.
- Channels 6–11 represent Black pieces.
- The board is not rotated when the DQN plays Black.

## Current training workflow

Running the training module:

1. Creates the environment, agent, opponent, and replay buffer.
2. Reads the previous best evaluation score from `best.pt` when available.
3. Loads `latest.pt` when available to resume training.
4. Runs multi-episode training.
5. Alternates the DQN between White and Black.
6. Stores rewards from the DQN's perspective.
7. Reports progress during training.
8. Synchronizes the target network periodically.
9. Saves `latest.pt` periodically.
10. Evaluates the current policy periodically.
11. Evaluates equally as White and Black.
12. Calculates a normalized evaluation score.
13. Replaces `best.pt` only when the new score is strictly better.
14. Prints an aggregated training summary.

## Current evaluation metrics

- Wins
- Draws
- Losses
- Truncated games
- Normalized evaluation score

Evaluation score is:

`(wins + 0.5 * draws) / episodes`

Periodic model-selection evaluation currently uses:

- 10 games as White
- 10 games as Black

## Current checkpoint roles

- `checkpoints/latest.pt` — most recent resumable training state.
- `checkpoints/best.pt` — best balanced evaluation score observed so far.

Training checkpoints can preserve:

- Policy network
- Target network
- Optimizer
- Epsilon
- Replay-buffer capacity
- Replay-buffer transitions
- Optional checkpoint metadata

## Tests

The repository currently defines 116 tests covering the environment,
encoding, DQN agent, replay buffer, episode execution, training workflow,
evaluation, checkpointing, reward perspective, and color alternation.

## Current limitations

- The opponent is still RandomAgent.
- Self-play is not implemented.
- Board encoding remains absolute rather than agent-relative.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not store a global lifetime episode counter.
- Checkpoint compatibility assumes compatible code and hyperparameters.
- Evaluation currently uses only RandomAgent as the benchmark.
- Evaluation scores may be noisy because they use a finite number of games.

## Current focus

The DQN-versus-RandomAgent stage now supports balanced White/Black
training and evaluation.

## Next milestone

Design the first self-play workflow.

Before implementation, determine how two DQN-controlled sides should
generate experience and how opponent-network updates should be managed.