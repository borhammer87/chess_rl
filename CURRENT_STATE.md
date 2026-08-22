# CURRENT STATE

## Status

The project currently contains a complete DQN training, evaluation, and
resumable checkpoint workflow.

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
- Periodic checkpointing
- Training checkpoint loading
- Replay-buffer persistence

## Training package structure

Training responsibilities have been separated into focused modules:

- `results.py` — training and evaluation result data structures.
- `episodes.py` — step, episode, and replay-training operations.
- `checkpoint.py` — resumable training-state persistence.
- `train_dqn.py` — multi-episode workflow, evaluation scheduling,
  summaries, and main program execution.

This refactor changed structure without changing training behavior.

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

## Current checkpoint state

- Policy network
- Target network
- Optimizer
- Epsilon
- Replay-buffer capacity
- Replay-buffer transitions

## Current workflow

Running the training module:

1. Creates the environment, agent, opponent, and replay buffer.
2. Loads `checkpoints/latest.pt` when available.
3. Runs multi-episode training.
4. Reports progress during training.
5. Synchronizes the target network periodically.
6. Saves `latest.pt` periodically through a checkpoint callback.
7. Evaluates the current policy periodically through an evaluation callback.
8. Prints an aggregated training summary.

## Current limitations

- The DQN currently plays only White.
- Black is always controlled by RandomAgent.
- Self-play is not implemented.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not store a global lifetime episode counter.
- Checkpoint compatibility currently assumes compatible code and
  hyperparameter configuration.
- Evaluation results are reported but are not yet used to select checkpoints.

## Current focus

Periodic evaluation is now integrated into the training workflow.

The next development step is to use evaluation results to distinguish
the latest training state from the best-performing policy.

## Next milestone

Define a model-selection criterion and implement `best.pt` checkpoint
selection using the existing evaluation workflow.