# ARCHITECTURE

## Core interaction flow

ChessEnv
→ BoardEncoder
→ Tensor
→ DQNCNN
→ 4096 Q-values
→ Legal Mask
→ decode_legal_action
→ ChessEnv.step
→ ReplayBuffer
→ train_step

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

Contains the lower-level episode interaction and learning operations:

- `run_single_step()`
- `run_and_store_step()`
- `train_from_replay()`
- `run_episode()`
- `run_dqn_vs_random_episode()`

This module answers the question:

> What happens inside an episode?

### `train_dqn.py`

Coordinates complete training runs.

Its responsibilities include:

- Multi-episode training.
- Progress callbacks.
- Target-network synchronization.
- Checkpoint scheduling.
- Evaluation scheduling.
- Training summaries.
- Main program execution.

This module answers the question:

> How is a complete training session coordinated?

### `checkpoint.py`

Composes and restores resumable training checkpoints.

A training checkpoint contains state owned by:

- `DQNAgent`
- `ReplayBuffer`

## Training workflow

`train_against_random()` coordinates multi-episode training.

The same agent and replay buffer are reused across episodes.

After configured numbers of completed episodes, the workflow can trigger:

- Progress reporting.
- Target-network synchronization.
- Checkpoint callbacks.
- Evaluation callbacks.

The DQN currently plays White and RandomAgent plays Black.

## Evaluation

Evaluation uses the existing DQN-versus-RandomAgent episode infrastructure.

During evaluation:

- The agent uses a greedy policy.
- Training is disabled.
- Wins, draws, losses, and truncated games are collected.
- The previous epsilon value is restored afterwards.

`main()` currently schedules evaluation periodically during training.

## Persistence

Persistence responsibilities are separated between components.

### DQNAgent

`DQNAgent` owns:

- Policy network
- Target network
- Optimizer
- Epsilon

It exposes:

- `state_dict()`
- `load_state_dict()`

### ReplayBuffer

`ReplayBuffer` owns:

- Capacity
- Stored transitions

It exposes:

- `state_dict()`
- `load_state_dict()`

### Training checkpoint

`checkpoint.py` combines the agent and replay-buffer states.

The training workflow decides when checkpoint callbacks occur.

`main()` decides where the checkpoint is stored.

Current checkpoint path:

`checkpoints/latest.pt`