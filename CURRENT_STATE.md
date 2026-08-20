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
- Periodic checkpointing
- Training checkpoint loading
- Replay-buffer persistence

Current training metrics include:

- Episode reward
- Training losses
- Final epsilon
- Replay buffer size

Current evaluation metrics include:

- Wins
- Draws
- Losses
- Truncated games

Current checkpoint state includes:

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
6. Saves `latest.pt` periodically through the checkpoint callback.
7. Prints an aggregated training summary.

## Current limitations

- The DQN currently plays only White.
- Black is always controlled by RandomAgent.
- Self-play is not implemented.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not store a global lifetime episode counter.
- Checkpoint compatibility currently assumes compatible code and
  hyperparameter configuration.

## Current focus

The basic long-running training workflow is now usable.

The next phase should use evaluation to compare trained policies and make
better decisions about which checkpoints are worth retaining.

## Next milestone

Integrate evaluation into the training workflow and prepare
evaluation-driven checkpoint selection, such as retaining a `best.pt`
checkpoint in addition to `latest.pt`.