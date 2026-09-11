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
- Generic opponent episode engine
- Frozen DQN self-play opponent
- Greedy frozen-opponent move selection
- Multi-episode self-play training
- Alternating White/Black self-play episodes
- Periodic frozen-opponent synchronization

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
- `episodes.py` — step, generic opponent episode, reward-perspective,
  and replay-training operations.
- `checkpoint.py` — training-state persistence and checkpoint metadata.
- `train_dqn.py` — multi-episode workflow, color alternation, evaluation
  scheduling, model selection, summaries, and main program execution.
- `self_play.py` — frozen DQN opponents, self-play episode wrappers,
  multi-episode self-play, and opponent-network synchronization.

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

1. Creates the environment, DQN learner, RandomAgent benchmark, and replay buffer.
2. Reads the previous best evaluation score from `best.pt` when available.
3. Loads `latest.pt` when available to resume training.
4. Creates a frozen training opponent from the restored learner `policy_net`.
5. Runs multi-episode self-play with `train_against_frozen()`.
6. Alternates the DQN learner between White and Black.
7. Stores rewards from the DQN's perspective.
8. Reports progress during training.
9. Synchronizes the target network every 10 episodes.
10. Saves `latest.pt` every 25 episodes.
11. Evaluates the current learner against RandomAgent every 25 episodes.
12. Evaluates equally as White and Black.
13. Calculates a normalized evaluation score.
14. Replaces `best.pt` only when the new score is strictly better.
15. Refreshes the frozen self-play opponent every 25 episodes.
16. Prints an aggregated training summary using the existing `TrainingSummary` infrastructure.

RandomAgent remains the provisional stable evaluation benchmark. It is used
for periodic evaluation and best-checkpoint selection, but it is no longer
the opponent used by the main training workflow.

Frozen-opponent synchronization is independent from target-network
synchronization. They currently use different frequencies and serve
different purposes.

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

The repository currently defines 170 tests covering the environment,
encoding, DQN agent, replay buffer, generic episode execution,
RandomAgent training, frozen-opponent self-play, evaluation,
checkpointing, reward perspective, and color alternation.

## Current limitations

- Self-play agents are not yet evaluated against frozen or historical
  policies.
- Board encoding remains absolute rather than agent-relative.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not store a global lifetime episode counter.
- Checkpoint compatibility assumes compatible code and hyperparameters.
- Evaluation currently uses only RandomAgent as the benchmark.
- Evaluation scores may be noisy because they use a finite number of games.

## Current focus

The first frozen-opponent self-play workflow is implemented and tested.

The learning DQN can train across multiple episodes while alternating
colors, periodically synchronizing both its target network and a separate
frozen opponent network.

## Next milestone

Run the integrated self-play workflow from the main program and inspect
real training behaviour and RandomAgent evaluation results.

Champion-vs-challenger evaluation and promotion criteria remain future work.