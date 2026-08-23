# CURRENT STATE

## Status

The project currently contains a complete DQN training, evaluation, and
checkpoint-selection workflow.

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
- Evaluation against RandomAgent
- Periodic evaluation during training
- Evaluation scoring
- Periodic checkpointing
- Training checkpoint loading
- Replay-buffer persistence
- Best-checkpoint selection

## Training package structure

Training responsibilities are separated into focused modules:

- `results.py` — training and evaluation result data structures.
- `episodes.py` — step, episode, and replay-training operations.
- `checkpoint.py` — training-state persistence and checkpoint metadata.
- `train_dqn.py` — multi-episode workflow, evaluation scheduling,
  model selection, summaries, and main program execution.

## Current training metrics

- Episode reward
- Training losses
- Final epsilon
- Replay buffer size

## Current evaluation metrics

- Wins
- Draws
- Losses
- Truncated games
- Normalized evaluation score

Evaluation score is calculated as:

`(wins + 0.5 * draws) / episodes`

Wins are worth 1 point, draws 0.5 points, and losses or truncated games
0 points.

## Current checkpoint state

Training checkpoints can preserve:

- Policy network
- Target network
- Optimizer
- Epsilon
- Replay-buffer capacity
- Replay-buffer transitions
- Optional checkpoint metadata

Two checkpoint roles currently exist:

- `checkpoints/latest.pt` — most recent resumable training state.
- `checkpoints/best.pt` — best evaluation score observed so far.

`best.pt` stores its evaluation score as checkpoint metadata so model
selection can continue correctly after restarting the program.

## Current workflow

Running the training module:

1. Creates the environment, agent, opponent, and replay buffer.
2. Reads the previous best evaluation score from `best.pt` when available.
3. Loads `latest.pt` when available to resume training.
4. Runs multi-episode training.
5. Reports progress during training.
6. Synchronizes the target network periodically.
7. Saves `latest.pt` periodically through a checkpoint callback.
8. Evaluates the current policy periodically.
9. Calculates a normalized evaluation score.
10. Replaces `best.pt` only when the new score is strictly better.
11. Prints an aggregated training summary.

## Current limitations

- The DQN currently plays only White.
- Black is always controlled by RandomAgent.
- Self-play is not implemented.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not store a global lifetime episode counter.
- Checkpoint compatibility currently assumes compatible code and
  hyperparameter configuration.
- Evaluation currently uses only RandomAgent as the benchmark.
- Evaluation scores may be noisy because they are calculated from a
  limited number of games.

## Current focus

The training, evaluation, persistence, and best-model selection workflow
is now complete for the current DQN-versus-RandomAgent stage.

## Next milestone

Decide how the agent should support both White and Black before moving
toward self-play.

This requires revisiting reward perspective and state representation.