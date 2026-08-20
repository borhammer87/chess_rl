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

## Training workflow

`train_against_random()` coordinates multi-episode training.

Its responsibilities include:

- Running DQN-versus-RandomAgent episodes.
- Reusing the same ReplayBuffer across episodes.
- Reporting progress through a callback.
- Synchronizing the target network periodically.
- Deciding when checkpoint callbacks are triggered.

The DQN currently plays White and RandomAgent plays Black.

## Persistence

Persistence responsibilities are separated between components.

### DQNAgent

`DQNAgent` owns its training state:

- Policy network
- Target network
- Optimizer
- Epsilon

It exposes:

- `state_dict()`
- `load_state_dict()`

### ReplayBuffer

`ReplayBuffer` owns its replay-memory state:

- Capacity
- Stored transitions

It exposes:

- `state_dict()`
- `load_state_dict()`

### Training layer

`train_dqn.py` composes the agent and replay-buffer states into a
training checkpoint.

The training workflow decides when a checkpoint should be created.

`main()` decides where the checkpoint is stored.

Current checkpoint path:

`checkpoints/latest.pt`